import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import axios, { AxiosResponse } from 'axios';

// MCP Protocol Types
interface MCPMessage {
  jsonrpc: '2.0';
  id?: string | number;
  method?: string;
  params?: any;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
  };
}

interface MCPResource {
  uri: string;
  name: string;
  description?: string;
  mimeType?: string;
}

interface MCPPrompt {
  name: string;
  description: string;
  arguments?: {
    name: string;
    description: string;
    required?: boolean;
  }[];
}

interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

interface OllamaRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  context?: number[];
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_predict?: number;
  };
}

// Configuration interfaces
interface MCPServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
}

interface OllamaConfig {
  baseUrl?: string;
  model: string;
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_predict?: number;
  };
}

interface MCPClientConfig {
  ollama: OllamaConfig;
  server: MCPServerConfig;
  timeout?: number;
  debug?: boolean;
}

class MCPClient extends EventEmitter {
  private serverProcess: ChildProcess | null = null;
  private messageId = 0;
  private pendingRequests = new Map<string | number, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>();
  private tools: MCPTool[] = [];
  private resources: MCPResource[] = [];
  private prompts: MCPPrompt[] = [];
  private isInitialized = false;
  private context: number[] = [];

  constructor(private config: MCPClientConfig) {
    super();
    this.setupErrorHandling();
  }

  private setupErrorHandling(): void {
    process.on('unhandledRejection', (reason, promise) => {
      this.emit('error', new Error(`Unhandled Rejection: ${reason}`));
    });

    process.on('uncaughtException', (error) => {
      this.emit('error', error);
    });
  }

  private log(message: string, data?: any): void {
    if (this.config.debug) {
      console.log(`[MCP Client] ${message}`, data || '');
    }
  }

  private generateId(): number {
    return ++this.messageId;
  }

  async initialize(): Promise<void> {
    try {
      this.log('Initializing MCP client...');
      
      // Start the MCP server process
      await this.startServer();
      
      // Initialize the MCP session
      await this.initializeSession();
      
      // Load available capabilities
      await this.loadCapabilities();
      
      this.isInitialized = true;
      this.log('MCP client initialized successfully');
      this.emit('ready');
    } catch (error) {
      this.log('Failed to initialize MCP client', error);
      throw error;
    }
  }

  private async startServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      const { command, args = [], env, cwd } = this.config.server;
      
      this.log(`Starting server: ${command} ${args.join(' ')}`);
      
      this.serverProcess = spawn(command, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, ...env },
        cwd: cwd || process.cwd()
      });

      if (!this.serverProcess.stdin || !this.serverProcess.stdout) {
        reject(new Error('Failed to create server process streams'));
        return;
      }

      // Handle server stdout (MCP messages)
      let buffer = '';
      this.serverProcess.stdout.on('data', (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.trim()) {
            try {
              const message: MCPMessage = JSON.parse(line);
              this.handleMessage(message);
            } catch (error) {
              this.log('Failed to parse message:', line);
            }
          }
        });
      });

      // Handle server stderr
      this.serverProcess.stderr?.on('data', (data: Buffer) => {
        this.log('Server stderr:', data.toString());
      });

      // Handle server exit
      this.serverProcess.on('exit', (code, signal) => {
        this.log(`Server process exited with code ${code}, signal ${signal}`);
        this.emit('serverExit', { code, signal });
      });

      this.serverProcess.on('error', (error) => {
        this.log('Server process error:', error);
        reject(error);
      });

      // Give the server a moment to start
      setTimeout(resolve, 1000);
    });
  }

  private async initializeSession(): Promise<void> {
    const initMessage: MCPMessage = {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {},
          resources: {},
          prompts: {}
        },
        clientInfo: {
          name: 'mcp-ollama-client',
          version: '1.0.0'
        }
      }
    };

    const response = await this.sendMessage(initMessage);
    this.log('Session initialized', response);
  }

  private async loadCapabilities(): Promise<void> {
    try {
      // Load tools
      const toolsResponse = await this.sendMessage({
        jsonrpc: '2.0',
        id: this.generateId(),
        method: 'tools/list'
      });
      this.tools = toolsResponse.tools || [];
      this.log(`Loaded ${this.tools.length} tools`);

      // Load resources
      const resourcesResponse = await this.sendMessage({
        jsonrpc: '2.0',
        id: this.generateId(),
        method: 'resources/list'
      });
      this.resources = resourcesResponse.resources || [];
      this.log(`Loaded ${this.resources.length} resources`);

      // Load prompts
      const promptsResponse = await this.sendMessage({
        jsonrpc: '2.0',
        id: this.generateId(),
        method: 'prompts/list'
      });
      this.prompts = promptsResponse.prompts || [];
      this.log(`Loaded ${this.prompts.length} prompts`);
    } catch (error) {
      this.log('Error loading capabilities:', error);
      // Don't fail initialization if capabilities loading fails
    }
  }

  private sendMessage(message: MCPMessage): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.serverProcess?.stdin) {
        reject(new Error('Server not running'));
        return;
      }

      const id = message.id || this.generateId();
      message.id = id;

      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request ${id} timed out`));
      }, this.config.timeout || 30000);

      this.pendingRequests.set(id, { resolve, reject, timeout });

      const messageStr = JSON.stringify(message) + '\n';
      this.serverProcess.stdin.write(messageStr);
      this.log('Sent message:', message);
    });
  }

  private handleMessage(message: MCPMessage): void {
    this.log('Received message:', message);

    if (message.id && this.pendingRequests.has(message.id)) {
      const { resolve, reject, timeout } = this.pendingRequests.get(message.id)!;
      this.pendingRequests.delete(message.id);
      clearTimeout(timeout);

      if (message.error) {
        reject(new Error(`MCP Error ${message.error.code}: ${message.error.message}`));
      } else {
        resolve(message.result);
      }
    } else if (message.method) {
      // Handle notifications or requests from server
      this.emit('notification', message);
    }
  }

  // Ollama integration methods
  private async callOllama(prompt: string, options?: Partial<OllamaRequest>): Promise<OllamaResponse> {
    const { baseUrl = 'http://localhost:11434', model } = this.config.ollama;
    
    const request: OllamaRequest = {
      model,
      prompt,
      stream: false,
      context: this.context,
      options: {
        ...this.config.ollama.options,
        ...options?.options
      },
      ...options
    };

    try {
      const response: AxiosResponse<OllamaResponse> = await axios.post(
        `${baseUrl}/api/generate`,
        request,
        {
          timeout: this.config.timeout || 30000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      // Update context for conversation continuity
      if (response.data.context) {
        this.context = response.data.context;
      }

      return response.data;
    } catch (error) {
      throw new Error(`Ollama API error: ${error}`);
    }
  }

  // Public API methods
  async chat(prompt: string, options?: {
    useTools?: boolean;
    maxToolCalls?: number;
    ollamaOptions?: Partial<OllamaRequest>;
  }): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Client not initialized. Call initialize() first.');
    }

    const { useTools = true, maxToolCalls = 5, ollamaOptions = {} } = options || {};
    let currentPrompt = prompt;
    let toolCallCount = 0;

    // Add available tools to the prompt if requested
    if (useTools && this.tools.length > 0) {
      const toolsInfo = this.tools.map(tool => 
        `Tool: ${tool.name}\nDescription: ${tool.description}\nInput Schema: ${JSON.stringify(tool.inputSchema, null, 2)}`
      ).join('\n\n');

      currentPrompt = `You have access to the following tools:\n\n${toolsInfo}\n\nTo use a tool, respond with JSON in this format:
{
  "tool_call": {
    "name": "tool_name",
    "arguments": { "param1": "value1" }
  }
}

User query: ${prompt}`;
    }

    while (toolCallCount < maxToolCalls) {
      const response = await this.callOllama(currentPrompt, ollamaOptions);
      const responseText = response.response.trim();

      // Check if the response contains a tool call
      if (useTools && this.isToolCall(responseText)) {
        try {
          const toolCall = this.parseToolCall(responseText);
          const toolResult = await this.callTool(toolCall.name, toolCall.arguments);
          
          currentPrompt = `Previous response: ${responseText}\nTool result: ${JSON.stringify(toolResult)}\nPlease provide a final response to the user based on this information.`;
          toolCallCount++;
          continue;
        } catch (error) {
          this.log('Tool call error:', error);
          return `I tried to use a tool but encountered an error: ${error}. Here's my direct response: ${responseText}`;
        }
      }

      return responseText;
    }

    return "I've reached the maximum number of tool calls. Please try rephrasing your request.";
  }

  private isToolCall(text: string): boolean {
    try {
      const parsed = JSON.parse(text);
      return parsed.tool_call && parsed.tool_call.name;
    } catch {
      return false;
    }
  }

  private parseToolCall(text: string): { name: string; arguments: any } {
    const parsed = JSON.parse(text);
    return {
      name: parsed.tool_call.name,
      arguments: parsed.tool_call.arguments || {}
    };
  }

  async callTool(name: string, arguments: any): Promise<any> {
    if (!this.isInitialized) {
      throw new Error('Client not initialized. Call initialize() first.');
    }

    const tool = this.tools.find(t => t.name === name);
    if (!tool) {
      throw new Error(`Tool '${name}' not found`);
    }

    const message: MCPMessage = {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'tools/call',
      params: {
        name,
        arguments
      }
    };

    return await this.sendMessage(message);
  }

  async getResource(uri: string): Promise<any> {
    if (!this.isInitialized) {
      throw new Error('Client not initialized. Call initialize() first.');
    }

    const message: MCPMessage = {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'resources/read',
      params: { uri }
    };

    return await this.sendMessage(message);
  }

  async getPrompt(name: string, arguments?: Record<string, any>): Promise<any> {
    if (!this.isInitialized) {
      throw new Error('Client not initialized. Call initialize() first.');
    }

    const message: MCPMessage = {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'prompts/get',
      params: {
        name,
        arguments: arguments || {}
      }
    };

    return await this.sendMessage(message);
  }

  // Getters for capabilities
  getTools(): MCPTool[] {
    return [...this.tools];
  }

  getResources(): MCPResource[] {
    return [...this.resources];
  }

  getPrompts(): MCPPrompt[] {
    return [...this.prompts];
  }

  // Utility methods
  clearContext(): void {
    this.context = [];
  }

  async close(): Promise<void> {
    this.log('Closing MCP client...');
    
    // Clear pending requests
    for (const [id, { reject, timeout }] of this.pendingRequests) {
      clearTimeout(timeout);
      reject(new Error('Client is closing'));
    }
    this.pendingRequests.clear();

    // Close server process
    if (this.serverProcess) {
      this.serverProcess.kill('SIGTERM');
      
      // Force kill after timeout
      setTimeout(() => {
        if (this.serverProcess && !this.serverProcess.killed) {
          this.serverProcess.kill('SIGKILL');
        }
      }, 5000);

      this.serverProcess = null;
    }

    this.isInitialized = false;
    this.emit('closed');
    this.log('MCP client closed');
  }
}

// Example usage and factory function
export class MCPClientBuilder {
  private config: Partial<MCPClientConfig> = {};

  static create(): MCPClientBuilder {
    return new MCPClientBuilder();
  }

  withOllama(config: OllamaConfig): MCPClientBuilder {
    this.config.ollama = config;
    return this;
  }

  withServer(config: MCPServerConfig): MCPClientBuilder {
    this.config.server = config;
    return this;
  }

  withTimeout(timeout: number): MCPClientBuilder {
    this.config.timeout = timeout;
    return this;
  }

  withDebug(debug: boolean = true): MCPClientBuilder {
    this.config.debug = debug;
    return this;
  }

  build(): MCPClient {
    if (!this.config.ollama) {
      throw new Error('Ollama configuration is required');
    }
    if (!this.config.server) {
      throw new Error('Server configuration is required');
    }

    return new MCPClient(this.config as MCPClientConfig);
  }
}

// Export types and main class
export {
  MCPClient,
  MCPTool,
  MCPResource,
  MCPPrompt,
  MCPClientConfig,
  OllamaConfig,
  MCPServerConfig,
  OllamaResponse,
  OllamaRequest
};

// Example usage:
/*
const client = MCPClientBuilder.create()
  .withOllama({
    baseUrl: 'http://localhost:11434',
    model: 'llama2',
    options: {
      temperature: 0.7
    }
  })
  .withServer({
    command: 'node',
    args: ['path/to/mcp-server.js']
  })
  .withTimeout(30000)
  .withDebug(true)
  .build();

// Initialize and use
await client.initialize();

// Chat with automatic tool usage
const response = await client.chat("What's the weather like today?", {
  useTools: true,
  maxToolCalls: 3
});

console.log(response);

// Call specific tools
const toolResult = await client.callTool('weather_lookup', {
  location: 'New York'
});

// Clean up
await client.close();
*/
