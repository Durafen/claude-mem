# Feature: Gemini CLI Authentication Support

## Goal
Add `CLAUDE_MEM_GEMINI_AUTH_METHOD: "cli"` option to use the local `gemini` CLI binary instead of API key authentication.

## Why
- `gemini` CLI is already installed and authenticated (uses Google Cloud auth)
- No API key management needed
- Consistent with how Claude provider works (`CLAUDE_MEM_CLAUDE_AUTH_METHOD: "cli"`)
- Free tier usage via CLI

## Current State
```json
// ~/.claude-mem/settings.json
{
  "CLAUDE_MEM_PROVIDER": "gemini",
  "CLAUDE_MEM_GEMINI_API_KEY": "",           // ← only option currently
  "CLAUDE_MEM_GEMINI_MODEL": "gemini-2.5-flash-lite"
}
```

## Proposed State
```json
{
  "CLAUDE_MEM_PROVIDER": "gemini",
  "CLAUDE_MEM_GEMINI_AUTH_METHOD": "cli",    // ← NEW: "cli" or "api_key"
  "CLAUDE_MEM_GEMINI_API_KEY": "",           // used only if auth_method = "api_key"
  "CLAUDE_MEM_GEMINI_MODEL": "gemini-2.5-flash-lite"
}
```

## Implementation Plan

### 1. Settings (settings-routes.ts or equivalent)
- Add `CLAUDE_MEM_GEMINI_AUTH_METHOD` to allowed settings
- Valid values: `"cli"`, `"api_key"` (default: `"api_key"` for backwards compat)

### 2. Create GeminiCliProvider (new file)
```typescript
// gemini-cli-provider.ts (pseudo-code)
import { spawn } from 'child_process';

export async function callGeminiCli(prompt: string, model: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const args = [
      '--model', model,
      '--output-format', 'json',
      prompt
    ];

    const proc = spawn('gemini', args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => stdout += data);
    proc.stderr.on('data', (data) => stderr += data);

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Gemini CLI failed: ${stderr}`));
      }
    });
  });
}
```

### 3. Update Generator/SDK to use CLI provider
In the generator code that calls the AI model:
```typescript
if (provider === 'gemini') {
  const authMethod = settings.get('CLAUDE_MEM_GEMINI_AUTH_METHOD') || 'api_key';

  if (authMethod === 'cli') {
    return callGeminiCli(prompt, model);
  } else {
    return callGeminiApi(prompt, model, apiKey);  // existing code
  }
}
```

### 4. Add CLI availability check
```typescript
import { which } from 'which';  // or use child_process

function isGeminiCliAvailable(): boolean {
  try {
    return !!which.sync('gemini');
  } catch {
    return false;
  }
}
```

### 5. Update settings validator
```typescript
if (settings.CLAUDE_MEM_GEMINI_AUTH_METHOD &&
    !['cli', 'api_key'].includes(settings.CLAUDE_MEM_GEMINI_AUTH_METHOD)) {
  return { valid: false, error: 'CLAUDE_MEM_GEMINI_AUTH_METHOD must be "cli" or "api_key"' };
}
```

## Files to Modify

1. **`src/settings/`** - Add new setting key
2. **`src/providers/`** (or equivalent) - Add GeminiCliProvider
3. **`src/generator/`** or **`src/sdk/`** - Update to conditionally use CLI
4. **`hooks/CLAUDE.md`** - Document new setting

## Reference Implementation
See `~/Python-Projects/ai-cli/ai_cli/providers/gemini.py` for a working Python implementation:
```python
class GeminiProvider(CLIProvider):
    name = "gemini"
    cli_name = "gemini"
    config = CLIConfig(
        base_cmd=["gemini"],
        model_args=["--model"],
        json_args=["--output-format", "json"],
        prompt_mode="arg",
    )
```

## Testing
1. Set `CLAUDE_MEM_GEMINI_AUTH_METHOD: "cli"`
2. Set `CLAUDE_MEM_PROVIDER: "gemini"`
3. Trigger a PostToolUse hook
4. Check logs for successful Gemini CLI invocation

## Gemini CLI Info
```bash
$ which gemini
/usr/local/bin/gemini

$ gemini --version
0.26.0

$ gemini --help
# shows available flags including --model, --output-format
```

## Fallback Behavior
If `auth_method: "cli"` but `gemini` binary not found:
- Log error with helpful message
- Fall back to API key if available
- Or fall back to OpenRouter/Claude
