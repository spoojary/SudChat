import express, { Request, Response } from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import { fetchUrlTool, fetchUrl } from './tools/fetchUrl.js';

const app = express();
const client = new Anthropic();

if (!process.env.ANTHROPIC_API_KEY) {
  console.warn('⚠️  ANTHROPIC_API_KEY is not set. Set it before starting the server.');
}

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:4173'] }));
app.use(express.json());

const MODELS = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
];

const VALID_MODEL_IDS = new Set(MODELS.map((m) => m.id));

// ── Tools ────────────────────────────────────────────────────────────────────
//
// This is what makes it an "agent": we give Claude a set of tools it can call.
// Claude decides WHEN to call them and WHAT arguments to pass.
// Each tool lives in its own file under src/tools/.
//
const TOOLS: Anthropic.Tool[] = [fetchUrlTool];

async function executeTool(name: string, input: unknown): Promise<string> {
  if (name === 'fetch_url') {
    const { url } = input as { url: string };
    return fetchUrl(url);
  }
  return `Unknown tool: ${name}`;
}

// ── Routes ───────────────────────────────────────────────────────────────────

app.get('/api/models', (_req: Request, res: Response) => {
  res.json(MODELS);
});

interface ChatBody {
  messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  model?: string;
  system?: string;
}

app.post('/api/chat', async (req: Request, res: Response) => {
  const { messages, model = 'claude-opus-4-6', system } = req.body as ChatBody;

  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: 'messages array is required' });
    return;
  }
  if (!VALID_MODEL_IDS.has(model)) {
    res.status(400).json({ error: `Invalid model: ${model}` });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (payload: object) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  try {
    // Build the message history for the agentic loop.
    // We keep appending to this array each iteration.
    const agentMessages: Anthropic.MessageParam[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // ── Agentic loop ────────────────────────────────────────────────────────
    //
    // Each iteration:
    //   1. Call Claude with the current message history + tools
    //   2. Stream any text deltas to the frontend
    //   3. If Claude called a tool → execute it, append result, loop again
    //   4. If Claude is done (end_turn) → break
    //
    while (true) {
      const stream = client.messages.stream({
        model,
        max_tokens: 4096,
        ...(system?.trim() ? { system: system.trim() } : {}),
        tools: TOOLS,
        messages: agentMessages,
      });

      // Stream text tokens to the frontend as they arrive
      for await (const event of stream) {
        if (
          event.type === 'content_block_delta' &&
          event.delta.type === 'text_delta'
        ) {
          send({ type: 'text', text: event.delta.text });
        }
      }

      // Get the full response once streaming is done
      const response = await stream.finalMessage();

      if (response.stop_reason === 'end_turn') {
        // Claude is done — exit the loop
        break;
      }

      if (response.stop_reason === 'tool_use') {
        // Claude wants to call one or more tools.
        // 1. Append the assistant's response (including tool_use blocks) to history
        agentMessages.push({ role: 'assistant', content: response.content });

        // 2. Execute each tool and collect results
        const toolResults: Anthropic.ToolResultBlockParam[] = [];

        for (const block of response.content) {
          if (block.type !== 'tool_use') continue;

          // Tell the frontend a tool is running (so it can show a spinner)
          send({ type: 'tool_call', tool: block.name, input: block.input });

          const result = await executeTool(block.name, block.input);

          // Tell the frontend the tool finished
          send({ type: 'tool_result', tool: block.name });

          toolResults.push({
            type: 'tool_result',
            tool_use_id: block.id,
            content: result,
          });
        }

        // 3. Feed the tool results back to Claude and loop again
        agentMessages.push({ role: 'user', content: toolResults });
      } else {
        // Unexpected stop reason — bail out
        break;
      }
    }

    send({ type: 'done' });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unexpected server error';
    send({ type: 'error', message });
  }

  res.end();
});

const PORT = Number(process.env.PORT ?? 3001);
app.listen(PORT, () => {
  console.log(`SudChat backend → http://localhost:${PORT}`);
});
