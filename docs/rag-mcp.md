# RAG and MCP in SudChat

## What is RAG?

RAG = **Retrieve** relevant text → **Augment** Claude's context with it → Claude **Generates** a grounded answer.

In SudChat's architecture, RAG plugs in at one of two points:

```
User message
    │
    ├─ Option A: pre-inject  ──► inject retrieved chunks into system/messages BEFORE the loop
    │
    └─ Option B: as a tool   ──► give Claude a search_docs tool, let it call it inside the loop
```

**Option B (tool) is the better fit** — it follows the same pattern as `fetch_url` and `run_code`, and Claude decides *when* and *what* to search.

---

## RAG Examples (built-in tools)

### 1. In-Memory Keyword Search

Good for small, stable corpora (API docs, README, policy docs). No dependencies.

**`backend/src/tools/searchDocs.ts`**
```typescript
import Anthropic from '@anthropic-ai/sdk';

const DOCS: { title: string; content: string }[] = [
  { title: 'README', content: '...' },
  { title: 'API Reference', content: '...' },
];

function search(query: string, topK = 3) {
  const q = query.toLowerCase();
  return DOCS
    .map(doc => ({
      ...doc,
      score: (doc.content.toLowerCase().match(new RegExp(q, 'g')) ?? []).length,
    }))
    .filter(d => d.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(d => `### ${d.title}\n${d.content.slice(0, 2000)}`);
}

export const searchDocsTool: Anthropic.Tool = {
  name: 'search_docs',
  description: 'Search the documentation corpus for relevant information. ' +
    'Use this before answering questions about the product.',
  input_schema: {
    type: 'object' as const,
    properties: {
      query: { type: 'string', description: 'Search query' },
    },
    required: ['query'],
  },
};

export function searchDocs(query: string): string {
  const results = search(query);
  if (results.length === 0) return 'No relevant documents found.';
  return `Found ${results.length} relevant sections:\n\n${results.join('\n\n---\n\n')}`;
}
```

Register in `server.ts` exactly like the other tools:
```typescript
import { searchDocsTool, searchDocs } from './tools/searchDocs.js';

const TOOLS = [fetchUrlTool, runCodeTool, searchDocsTool];

// inside executeTool():
if (name === 'search_docs') {
  const { query } = input as { query: string };
  return { content: searchDocs(query) };
}
```

---

### 2. Semantic Vector Search

Good for large corpora where keyword matching misses synonyms. Uses Anthropic's embeddings API.

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic();

interface Chunk { text: string; embedding: number[]; source: string; }
const index: Chunk[] = [];

async function indexDocument(text: string, source: string) {
  const chunks = text.match(/.{1,500}/gs) ?? [];

  const { embeddings } = await client.embeddings.create({
    model: 'claude-embedding-001',
    texts: chunks,
  });

  for (let i = 0; i < chunks.length; i++) {
    index.push({ text: chunks[i], embedding: embeddings[i].embedding, source });
  }
}

function cosineSim(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (magA * magB);
}

export async function semanticSearch(query: string, topK = 4): Promise<string> {
  const { embeddings } = await client.embeddings.create({
    model: 'claude-embedding-001',
    texts: [query],
  });
  const qEmb = embeddings[0].embedding;

  const results = index
    .map(chunk => ({ ...chunk, score: cosineSim(qEmb, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return results.map(r => `[${r.source}]\n${r.text}`).join('\n\n---\n\n');
}
```

The tool definition stays the same as Example 1 — only the search implementation changes.

---

### 3. Pre-Inject RAG (no tool)

Good for always-relevant context: user profile, account data, session notes. Claude doesn't need to decide to search.

```typescript
// In server.ts, before the agentic loop:
const latestQuery = messages.at(-1)?.content ?? '';
const retrieved = await searchDocs(latestQuery);

const augmentedSystem = [
  system?.trim(),
  retrieved
    ? `Relevant documentation:\n<docs>\n${retrieved}\n</docs>`
    : null,
].filter(Boolean).join('\n\n');

// Pass augmentedSystem instead of system to client.messages.stream(...)
```

| | Tool (Option B) | Pre-inject (Option 3) |
|---|---|---|
| Claude controls retrieval | Yes | No |
| Retrieved every request | Only when needed | Always |
| Best for | Exploratory Q&A | Always-relevant context |

---

## RAG via MCP Server

MCP (Model Context Protocol) lets you move the RAG logic into a **separate, reusable server process**. SudChat connects to it as a client.

```
SudChat Frontend
      │
SudChat Backend (Express + agentic loop)
      │
      ├── [built-in tools]  fetch_url, run_code
      │
      └── [MCP client]  ──connects to──► RAG MCP Server
                                              │
                                         Document index
                                         (embeddings / vector DB)
```

Claude sees `search_docs` as just another tool — it doesn't know or care that it lives in a separate process.

---

### The RAG MCP Server

A standalone process that owns the document index.

**`rag-server/server.ts`**
```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

const DOCS = [
  { text: 'Anthropic rate limits: Tier 1 allows 50 RPM...', source: 'api-docs' },
  { text: 'Claude tool use: pass an input_schema...', source: 'api-docs' },
];

function keywordSearch(query: string, topK = 4): string {
  const q = query.toLowerCase();
  return DOCS
    .map(d => ({ ...d, score: (d.text.toLowerCase().match(new RegExp(q, 'g')) ?? []).length }))
    .filter(d => d.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(d => `[${d.source}] ${d.text}`)
    .join('\n\n---\n\n') || 'No results found.';
}

const server = new Server(
  { name: 'rag-server', version: '1.0.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: 'search_docs',
    description: 'Search the internal documentation corpus for relevant information.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
      },
      required: ['query'],
    },
  }],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name === 'search_docs') {
    const { query } = req.params.arguments as { query: string };
    return { content: [{ type: 'text', text: keywordSearch(query) }] };
  }
  return { content: [{ type: 'text', text: 'Unknown tool' }], isError: true };
});

// stdio transport — SudChat spawns this process
const transport = new StdioServerTransport();
await server.connect(transport);
```

Run with: `node rag-server/server.js`

---

### Connecting SudChat to the MCP Server

Two approaches:

#### Approach A — Anthropic SDK beta (fewest code changes)

The Anthropic API manages the MCP server connection server-side:

```typescript
// In server.ts — replace client.messages.stream(...) with:
const stream = await client.beta.messages.stream({
  model,
  max_tokens: 8096,
  messages: agentMessages,
  tools: TOOLS,                     // existing fetch_url, run_code
  mcp_servers: [{
    type: 'stdio',
    command: 'node',
    args: ['./rag-server/server.js'],
    // remote: { type: 'url', url: 'https://my-rag-server.com/mcp' }
  }],
  betas: ['mcp-client-2025-04-04'],
});
```

Anthropic's infrastructure spawns and manages the MCP server and routes tool calls automatically.

> **Downside:** beta feature; you lose the custom SSE `tool_result` meta currently forwarded to the frontend.

---

#### Approach B — Manual MCP client bridge (full control)

Keeps the existing agentic loop intact. Merges MCP tools into the tool list at request time.

**`backend/src/mcp.ts`**
```typescript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import Anthropic from '@anthropic-ai/sdk';

let mcpClient: Client | null = null;

export async function connectMcp() {
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['./rag-server/server.js'],
  });
  mcpClient = new Client({ name: 'sudchat-backend', version: '1.0.0' });
  await mcpClient.connect(transport);
  console.log('MCP RAG server connected');
}

export async function getMcpTools(): Promise<Anthropic.Tool[]> {
  if (!mcpClient) return [];
  const { tools } = await mcpClient.listTools();
  return tools.map(t => ({
    name: t.name,
    description: t.description ?? '',
    input_schema: t.inputSchema as Anthropic.Tool['input_schema'],
  }));
}

export async function callMcpTool(name: string, input: unknown): Promise<string> {
  if (!mcpClient) return 'MCP not connected';
  const result = await mcpClient.callTool({
    name,
    arguments: input as Record<string, unknown>,
  });
  return result.content
    .filter(c => c.type === 'text')
    .map(c => c.text)
    .join('\n');
}
```

**`backend/src/server.ts`** — minimal changes:
```typescript
import { connectMcp, getMcpTools, callMcpTool } from './mcp.js';

// Connect at startup
await connectMcp();

// In /api/chat — merge MCP tools with built-ins:
const mcpTools = await getMcpTools();
const allTools = [...TOOLS, ...mcpTools];   // [fetch_url, run_code, search_docs]

const stream = client.messages.stream({
  model,
  tools: allTools,
  messages: agentMessages,
  max_tokens: 8096,
});

// In executeTool() — fallback unknown tools to MCP:
async function executeTool(name: string, input: unknown): Promise<ToolResult> {
  if (name === 'fetch_url') { /* ... */ }
  if (name === 'run_code')  { /* ... */ }

  // Route anything else to the MCP server
  const content = await callMcpTool(name, input);
  return { content };
}
```

The frontend renders a "Searching docs…" activity row automatically — `tool_call`/`tool_result` SSE events fire the same way as for any other tool.

---

## Why MCP for RAG?

| | Built-in tool (`searchDocs.ts`) | MCP RAG server |
|---|---|---|
| Lives in | SudChat repo | Its own repo/process |
| Language | TypeScript | **Anything** (Python has better ML libs) |
| Reusable by | SudChat only | Any MCP-compatible client |
| Hot-swap index | Restart SudChat | Restart MCP server only |
| Scale separately | No | Yes |

The biggest win: write the RAG server in **Python** using `sentence-transformers` and `chromadb` for proper semantic embeddings, while SudChat stays in TypeScript. They communicate over stdio or HTTP.

```
rag-server/
  server.py          ← Python: chromadb + sentence-transformers

sudchat/
  backend/src/
    mcp.ts           ← TypeScript MCP client
    server.ts        ← unchanged agentic loop
```

### Transport options

| Transport | Use case |
|---|---|
| `StdioServerTransport` | Local process, same machine |
| HTTP / SSE | Remote server, separate host |
