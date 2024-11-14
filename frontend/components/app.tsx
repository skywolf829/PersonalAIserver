import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, Terminal, Image, Loader2, Send, LucideIcon } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ModelInterfaceProps } from '@/types';
import { Message } from '@/types';

// Replace with your public IP or domain
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface LoginProps {
  onLogin: (username: string, password: string) => Promise<void>;
  error: string | null;
}

const Login = ({ onLogin, error }: LoginProps) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await onLogin(username, password);
      console.log('Login successful');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-md p-8 space-y-6">
        <div className="space-y-2 text-center">
          <h1 className="text-3xl font-bold">Welcome Back</h1>
          <p className="text-gray-500">Sign in to access Skylars AI models</p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          
          {error && (
            <Alert variant={error === "Failed to fetch" ? "default" : "destructive"}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {error === "Failed to fetch" 
                  ? "The server appears to be down at the moment. Please try again later."
                  : error}
              </AlertDescription>
            </Alert>
          )}
          
          <Button 
            type="submit" 
            className="w-full"
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              'Sign In'
            )}
          </Button>
        </form>
      </Card>
    </div>
  );
};

interface ModelTileProps {
  icon: LucideIcon;
  title: string;
  description: string;
  onClick: () => void;
}

const ModelTile = ({ icon: Icon, title, description, onClick }: ModelTileProps) => (
  <Card 
    className="p-6 cursor-pointer hover:shadow-lg transition-shadow"
    onClick={onClick}
  >
    <div className="space-y-4">
      <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
        <Icon className="h-6 w-6 text-primary" />
      </div>
      <div>
        <h3 className="font-semibold">{title}</h3>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
    </div>
  </Card>
);

const ModelInterface: React.FC<ModelInterfaceProps> = ({ type, onBack }) => {
  const [prompt, setPrompt] = useState('');
  const [loraWeights, setLoraWeights] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: prompt,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);
    
    // Clear input while waiting
    setPrompt('');
    
    try {
      const response = await fetch(`${API_URL}/generate/${type}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ 
          prompt: userMessage.content,
          lora_weights: loraWeights
        })
      });
      
      if (!response.ok) throw new Error('Generation failed');
      
      const data = await response.json();

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.generated_content,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  if (type === 'image') {
    return (
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        <Button variant="ghost" onClick={onBack} className="mb-4">
          ← Back to Models
        </Button>
        
        <Card className="p-6 space-y-4">
          <h2 className="text-2xl font-bold">Image Generation</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Image Description</label>
              <Textarea
                placeholder="Describe the image you want to generate..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="min-h-[100px]"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">LoRA Weights (Optional)</label>
              <Input
                placeholder="Enter LoRA weights path..."
                value={loraWeights}
                onChange={(e) => setLoraWeights(e.target.value)}
              />
            </div>
            
            <Button type="submit" disabled={isLoading || !prompt} className="w-full">
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Send className="h-4 w-4 mr-2" />
              )}
              Generate
            </Button>
          </form>
          
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {messages.length > 0 && (
            <div className="space-y-4">
              {messages.map((msg, index) => (
                msg.role === 'assistant' && (
                  <div key={index} className="mt-4">
                    <img 
                      src={`data:image/png;base64,${msg.content}`}
                      alt="Generated content"
                      className="max-w-full rounded-lg shadow-lg"
                    />
                  </div>
                )
              ))}
            </div>
          )}
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <Button variant="ghost" onClick={onBack} className="mb-4">
        ← Back to Models
      </Button>
      
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Chat with AI</h2>
        
        <div className="space-y-4">
          {/* Messages Container */}
          <div className="space-y-4 mb-4 max-h-[60vh] overflow-y-auto">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground ml-4'
                      : 'bg-muted mr-4'
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">{message.content}</p>
                  <p className="text-xs mt-1 opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              placeholder="Type your message..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !prompt}>
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
};

const Dashboard = ({ onLogout }: { onLogout: () => void }) => {
  const [selectedModel, setSelectedModel] = useState<'text' | 'image' | null>(null);
  
  const models = [
    {
      type: 'text' as const,
      icon: Terminal,
      title: 'Text Generation',
      description: 'Generate text using LLaMA 3.2'
    },
    {
      type: 'image' as const,
      icon: Image,
      title: 'Image Generation',
      description: 'Create images using Stable Diffusion 3.5'
    }
  ] as const;

  if (selectedModel) {
    return (
      <ModelInterface 
        type={selectedModel}
        onBack={() => setSelectedModel(null)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">AI Models Dashboard</h1>
          <Button variant="outline" onClick={onLogout}>Sign Out</Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((model) => (
            <ModelTile
              key={model.type}
              {...model}
              onClick={() => setSelectedModel(model.type)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (username: string, password: string) => {
    try {
      const response = await fetch(`${API_URL}/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
      });

      if (!response.ok) {
        throw new Error('Invalid credentials');
      }

      const data = await response.json();
      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
  };

  return token ? (
    <Dashboard onLogout={handleLogout} />
  ) : (
    <Login onLogin={handleLogin} error={error} />
  );
};

export default App;