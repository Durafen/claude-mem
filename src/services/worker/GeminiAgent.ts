/**
 * GeminiAgent: Gemini-based observation extraction
 *
 * Alternative to SDKAgent that uses Google's Gemini API directly
 * for extracting observations from tool usage.
 *
 * Responsibility:
 * - Call Gemini REST API for observation extraction
 * - Parse XML responses (same format as Claude)
 * - Sync to database and Chroma
 */

import path from 'path';
import { homedir } from 'os';
import { execSync, spawn, type ChildProcess } from 'child_process';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { logger } from '../../utils/logger.js';
import { buildInitPrompt, buildObservationPrompt, buildSummaryPrompt, buildContinuationPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { getCredential } from '../../shared/EnvManager.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { ModeManager } from '../domain/ModeManager.js';
import {
  processAgentResponse,
  shouldFallbackToClaude,
  isAbortError,
  type WorkerRef,
  type FallbackAgent
} from './agents/index.js';

// Gemini API endpoint
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

// CLI timeout for subprocess execution (2 minutes)
const DEFAULT_CLI_TIMEOUT = 120000;

// Gemini model types (available via API)
export type GeminiModel =
  | 'gemini-2.5-flash-lite'
  | 'gemini-2.5-flash'
  | 'gemini-2.5-pro'
  | 'gemini-2.0-flash'
  | 'gemini-2.0-flash-lite'
  | 'gemini-3-flash-preview';

// Free tier RPM limits by model (requests per minute)
const GEMINI_RPM_LIMITS: Record<GeminiModel, number> = {
  'gemini-2.5-flash-lite': 10,
  'gemini-2.5-flash': 10,
  'gemini-2.5-pro': 5,
  'gemini-2.0-flash': 15,
  'gemini-2.0-flash-lite': 30,
  'gemini-3-flash-preview': 5,
};

// Track last request time for rate limiting
let lastRequestTime = 0;

/**
 * Enforce RPM rate limit for Gemini free tier.
 * Waits the required time between requests based on model's RPM limit + 100ms safety buffer.
 * Skipped entirely if rate limiting is disabled (billing users with 1000+ RPM available).
 */
async function enforceRateLimitForModel(model: GeminiModel, rateLimitingEnabled: boolean): Promise<void> {
  // Skip rate limiting if disabled (billing users with 1000+ RPM)
  if (!rateLimitingEnabled) {
    return;
  }

  const rpm = GEMINI_RPM_LIMITS[model] || 5;
  const minimumDelayMs = Math.ceil(60000 / rpm) + 100; // (60s / RPM) + 100ms safety buffer

  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;

  if (timeSinceLastRequest < minimumDelayMs) {
    const waitTime = minimumDelayMs - timeSinceLastRequest;
    logger.debug('SDK', `Rate limiting: waiting ${waitTime}ms before Gemini request`, { model, rpm });
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }

  lastRequestTime = Date.now();
}

interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
      }>;
    };
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}

/**
 * Gemini content message format
 * role: "user" or "model" (Gemini uses "model" not "assistant")
 */
interface GeminiContent {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

export class GeminiAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the fallback agent (Claude SDK) for when Gemini API fails
   * Must be set after construction to avoid circular dependency
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start Gemini agent for a session
   * Uses multi-turn conversation to maintain context across messages
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      // Get Gemini configuration
      const { apiKey, model, rateLimitingEnabled, authMethod } = this.getGeminiConfig();

      // Validate configuration based on auth method
      if (authMethod === 'api_key' && !apiKey) {
        throw new Error('Gemini API key not configured. Set CLAUDE_MEM_GEMINI_API_KEY in settings or GEMINI_API_KEY environment variable.');
      }
      if (authMethod === 'cli' && !this.findGeminiExecutable()) {
        throw new Error('Gemini CLI not found. Install from https://github.com/google-gemini/gemini-cli');
      }

      logger.info('SDK', `Starting Gemini agent (auth: ${authMethod}, model: ${model})`, { sessionDbId: session.sessionDbId });

      // Generate synthetic memorySessionId (Gemini is stateless, doesn't return session IDs)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `gemini-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=Gemini`);
      }

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query Gemini
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const geminiConfig = { apiKey, model, rateLimitingEnabled, authMethod };
      const initResponse = await this.queryGemini(session.conversationHistory, geminiConfig);

      if (initResponse.content) {
        // Add response to conversation history
        session.conversationHistory.push({ role: 'assistant', content: initResponse.content });

        // Track token usage
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);  // Rough estimate
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

        // Process response using shared ResponseProcessor (no original timestamp for init - not from queue)
        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          'Gemini'
        );
      } else {
        logger.error('SDK', 'Empty Gemini init response - session may lack context', {
          sessionId: session.sessionDbId,
          model
        });
      }

      // Process pending messages
      // Track cwd from messages for CLAUDE.md generation
      let lastCwd: string | undefined;

      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        // CLAIM-CONFIRM: Track message ID for confirmProcessed() after successful storage
        // The message is now in 'processing' status in DB until ResponseProcessor calls confirmProcessed()
        session.processingMessageIds.push(message._persistentId);

        // Capture cwd from each message for worktree support
        if (message.cwd) {
          lastCwd = message.cwd;
        }
        // Capture earliest timestamp BEFORE processing (will be cleared after)
        // This ensures backlog messages get their original timestamps, not current time
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          // Update last prompt number
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          // This prevents wasting tokens when we won't be able to store the result anyway
          if (!session.memorySessionId) {
            throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build observation prompt
          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          // Add to conversation history and query Gemini
          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryGemini(session.conversationHistory, geminiConfig);

          let tokensUsed = 0;
          if (obsResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });

            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'Gemini',
            lastCwd
          );

        } else if (message.type === 'summarize') {
          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          if (!session.memorySessionId) {
            throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build summary prompt
          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          // Add to conversation history and query Gemini
          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryGemini(session.conversationHistory, geminiConfig);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });

            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'Gemini',
            lastCwd
          );
        }
      }

      // Mark session complete
      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', 'Gemini agent completed', {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', 'Gemini agent aborted', { sessionId: session.sessionDbId });
        throw error;
      }

      // Check if we should fall back to Claude
      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', 'Gemini API failed, falling back to Claude SDK', {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        // Fall back to Claude - it will use the same session with shared conversationHistory
        // Note: With claim-and-delete queue pattern, messages are already deleted on claim
        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', 'Gemini agent error', { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  /**
   * Convert shared ConversationMessage array to Gemini's contents format
   * Maps 'assistant' role to 'model' for Gemini API compatibility
   */
  private conversationToGeminiContents(history: ConversationMessage[]): GeminiContent[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }]
    }));
  }

  /**
   * Query Gemini via REST API with full conversation history (multi-turn)
   * Sends the entire conversation context for coherent responses
   */
  private async queryGeminiMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: GeminiModel,
    rateLimitingEnabled: boolean
  ): Promise<{ content: string; tokensUsed?: number }> {
    const contents = this.conversationToGeminiContents(history);
    const totalChars = history.reduce((sum, m) => sum + m.content.length, 0);

    logger.debug('SDK', `Querying Gemini multi-turn (${model})`, {
      turns: history.length,
      totalChars
    });

    const url = `${GEMINI_API_URL}/${model}:generateContent?key=${apiKey}`;

    // Enforce RPM rate limit for free tier (skipped if rate limiting disabled)
    await enforceRateLimitForModel(model, rateLimitingEnabled);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.3,  // Lower temperature for structured extraction
          maxOutputTokens: 4096,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini API error: ${response.status} - ${error}`);
    }

    const data = await response.json() as GeminiResponse;

    if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
      logger.error('SDK', 'Empty response from Gemini');
      return { content: '' };
    }

    const content = data.candidates[0].content.parts[0].text;
    const tokensUsed = data.usageMetadata?.totalTokenCount;

    return { content, tokensUsed };
  }

  /**
   * Get Gemini configuration from settings or environment
   * Issue #733: Uses centralized ~/.claude-mem/.env for credentials, not random project .env files
   */
  private getGeminiConfig(): { apiKey: string; model: GeminiModel; rateLimitingEnabled: boolean; authMethod: 'cli' | 'api_key' } {
    const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

    // Auth method: 'cli' uses gemini CLI binary, 'api_key' uses REST API
    const authMethod = (settings.CLAUDE_MEM_GEMINI_AUTH_METHOD || 'api_key') as 'cli' | 'api_key';

    // API key: check settings first, then centralized claude-mem .env (NOT process.env)
    // This prevents Issue #733 where random project .env files could interfere
    const apiKey = settings.CLAUDE_MEM_GEMINI_API_KEY || getCredential('GEMINI_API_KEY') || '';

    // Model: from settings or default, with validation
    const defaultModel: GeminiModel = 'gemini-2.5-flash';
    const configuredModel = settings.CLAUDE_MEM_GEMINI_MODEL || defaultModel;
    const validModels: GeminiModel[] = [
      'gemini-2.5-flash-lite',
      'gemini-2.5-flash',
      'gemini-2.5-pro',
      'gemini-2.0-flash',
      'gemini-2.0-flash-lite',
      'gemini-3-flash-preview',
    ];

    let model: GeminiModel;
    if (validModels.includes(configuredModel as GeminiModel)) {
      model = configuredModel as GeminiModel;
    } else {
      logger.warn('SDK', `Invalid Gemini model "${configuredModel}", falling back to ${defaultModel}`, {
        configured: configuredModel,
        validModels,
      });
      model = defaultModel;
    }

    // Rate limiting: enabled by default for free tier users
    const rateLimitingEnabled = settings.CLAUDE_MEM_GEMINI_RATE_LIMITING_ENABLED !== 'false';

    return { apiKey, model, rateLimitingEnabled, authMethod };
  }

  /**
   * Find gemini CLI executable path
   * Uses 'which' on Unix or 'where' on Windows
   */
  private findGeminiExecutable(): string | null {
    try {
      const cmd = process.platform === 'win32' ? 'where gemini' : 'which gemini';
      const result = execSync(cmd, {
        encoding: 'utf8',
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore']
      }).trim().split('\n')[0].trim();
      return result || null;
    } catch {
      return null;
    }
  }

  /**
   * Query Gemini via CLI subprocess
   * Command: gemini --model <model> "<prompt>"
   */
  private async queryGeminiCli(prompt: string, model: string): Promise<{ content: string }> {
    return new Promise((resolve, reject) => {
      const geminiPath = this.findGeminiExecutable();
      if (!geminiPath) {
        reject(new Error('Gemini CLI not found. Install from https://github.com/google-gemini/gemini-cli'));
        return;
      }

      logger.debug('SDK', `Querying Gemini CLI (${model})`, { promptLength: prompt.length });

      // Command: gemini --model <model> -p "<prompt>"
      // -p flag is required for non-interactive (headless) mode
      const args = ['--model', model, '-p', prompt];
      const proc = spawn(geminiPath, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });

      let stdout = '';
      let stderr = '';
      let timeoutHandle: NodeJS.Timeout | null = null;

      // Set timeout
      timeoutHandle = setTimeout(() => {
        proc.kill('SIGTERM');
        reject(new Error(`Gemini CLI timed out after ${DEFAULT_CLI_TIMEOUT / 1000}s`));
      }, DEFAULT_CLI_TIMEOUT);

      proc.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
      proc.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

      proc.on('close', (code: number | null) => {
        if (timeoutHandle) clearTimeout(timeoutHandle);

        if (code !== 0) {
          logger.error('SDK', `Gemini CLI failed (exit ${code})`, { stderr: stderr.slice(0, 500) });
          reject(new Error(`Gemini CLI failed (exit ${code}): ${stderr.slice(0, 200)}`));
        } else {
          logger.debug('SDK', 'Gemini CLI response received', { responseLength: stdout.length });
          resolve({ content: stdout.trim() });
        }
      });

      proc.on('error', (err: Error) => {
        if (timeoutHandle) clearTimeout(timeoutHandle);
        reject(err);
      });
    });
  }

  /**
   * Unified query method - routes to CLI or API based on config
   * For CLI mode: uses only the last message (CLI is stateless)
   * For API mode: uses full conversation history
   */
  private async queryGemini(
    history: ConversationMessage[],
    config: { apiKey: string; model: GeminiModel; rateLimitingEnabled: boolean; authMethod: 'cli' | 'api_key' }
  ): Promise<{ content: string; tokensUsed?: number }> {
    const { apiKey, model, rateLimitingEnabled, authMethod } = config;

    if (authMethod === 'cli') {
      // CLI mode: use only the last message (CLI is stateless, no multi-turn)
      const lastMessage = history[history.length - 1];
      if (!lastMessage) {
        return { content: '' };
      }
      const response = await this.queryGeminiCli(lastMessage.content, model);
      return { content: response.content, tokensUsed: undefined };
    } else {
      // API mode: use full conversation history
      return this.queryGeminiMultiTurn(history, apiKey, model, rateLimitingEnabled);
    }
  }
}

/**
 * Check if Gemini is available based on configured auth method
 * - 'cli': checks if gemini CLI exists
 * - 'api_key': checks for API key in settings or env
 * Issue #733: Uses centralized ~/.claude-mem/.env, not random project .env files
 */
export function isGeminiAvailable(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  const authMethod = settings.CLAUDE_MEM_GEMINI_AUTH_METHOD || 'api_key';

  if (authMethod === 'cli') {
    // Check if gemini CLI exists
    try {
      const cmd = process.platform === 'win32' ? 'where gemini' : 'which gemini';
      execSync(cmd, { stdio: 'ignore', windowsHide: true });
      return true;
    } catch {
      return false;
    }
  }

  // API key mode (default)
  return !!(settings.CLAUDE_MEM_GEMINI_API_KEY || getCredential('GEMINI_API_KEY'));
}

/**
 * Check if Gemini is the selected provider
 */
export function isGeminiSelected(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'gemini';
}
