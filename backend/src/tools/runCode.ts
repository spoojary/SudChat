import Anthropic from '@anthropic-ai/sdk';
import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFile, unlink } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';

const execAsync = promisify(exec);

// ── Tool definition (sent to Claude so it knows the tool exists) ──────────────

export const runCodeTool: Anthropic.Tool = {
  name: 'run_code',
  description:
    'Execute Python or JavaScript code and return its output. ' +
    'Use this to test code you write, verify correctness, debug errors, ' +
    'and iterate until the solution works. Always run code before presenting it as a final answer.',
  input_schema: {
    type: 'object' as const,
    properties: {
      code: {
        type: 'string',
        description: 'The complete, self-contained code to execute',
      },
      language: {
        type: 'string',
        enum: ['python', 'javascript'],
        description: 'Programming language: "python" or "javascript"',
      },
    },
    required: ['code', 'language'],
  },
};

// ── Result type (returned to both Claude and the frontend) ────────────────────

export interface CodeRunResult {
  exitCode: number;
  stdout: string;
  stderr: string;
  language: string;
}

// ── Tool implementation ───────────────────────────────────────────────────────

export async function runCode(
  code: string,
  language: 'python' | 'javascript',
): Promise<CodeRunResult> {
  const ext = language === 'python' ? 'py' : 'js';
  const filepath = join(tmpdir(), `sudchat_${randomUUID()}.${ext}`);

  try {
    await writeFile(filepath, code, 'utf8');

    const cmd = language === 'python'
      ? `python3 "${filepath}"`
      : `node "${filepath}"`;

    const { stdout, stderr } = await execAsync(cmd, {
      timeout: 10_000,
      maxBuffer: 1024 * 512, // 512 KB output limit
    });

    return {
      exitCode: 0,
      stdout: stdout.trim(),
      stderr: stderr.trim(),
      language,
    };
  } catch (err: unknown) {
    const e = err as { stdout?: string; stderr?: string; code?: number; killed?: boolean };
    return {
      exitCode: typeof e.code === 'number' ? e.code : 1,
      stdout: e.stdout?.trim() ?? '',
      stderr: e.killed
        ? 'Process timed out after 10 seconds'
        : (e.stderr?.trim() ?? String(err)),
      language,
    };
  } finally {
    await unlink(filepath).catch(() => {});
  }
}

// ── Format result as a string for Claude ─────────────────────────────────────
// Claude receives this as the tool_result content so it can react to
// errors, re-try, or report success.

export function formatCodeResult(result: CodeRunResult): string {
  const parts = [`Exit code: ${result.exitCode}`];
  if (result.stdout) parts.push(`\nOutput:\n${result.stdout}`);
  if (result.stderr) parts.push(`\nErrors:\n${result.stderr}`);
  return parts.join('\n');
}
