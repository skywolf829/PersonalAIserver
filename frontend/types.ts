// types.ts
import { LucideIcon } from 'lucide-react'

export interface LoginProps {
  onLogin: (username: string, password: string) => Promise<void>
  error: string | null
}

export interface ModelTileProps {
  icon: LucideIcon
  title: string
  description: string
  onClick: () => void
}

export interface ModelInterfaceProps {
  type: 'text' | 'image' | 'video' | '3d'
  onBack: () => void
}

export interface DashboardProps {
  onLogout: () => void
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface QueueResponse {
  status: 'queued';
  position: number;
  message: string;
}