import React, { useState, useEffect, useMemo } from 'react';
import ReactFlow, { 
  Background, 
  Controls, 
  Panel,
  useNodesState,
  useEdgesState,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Brain, 
  ShieldCheck, 
  AlertTriangle, 
  Clock, 
  Cpu, 
  Database, 
  Globe,
  Terminal,
  Zap
} from 'lucide-react';

// --- Mock Data (Replace with real-time WebSocket data later) ---
const SERVICES = [
  { id: 'web-gateway', type: 'frontend', label: 'Web Gateway' },
  { id: 'cart-service', type: 'app', label: 'Cart Service' },
  { id: 'order-service', type: 'app', label: 'Order Service' },
  { id: 'product-catalog', type: 'app', label: 'Catalog' },
  { id: 'postgres-primary', type: 'data', label: 'Postgres DB' },
  { id: 'redis-cache', type: 'data', label: 'Redis' },
  { id: 'api-gateway', type: 'infra', label: 'API Gateway' },
];

const INITIAL_NODES = SERVICES.map((s, i) => ({
  id: s.id,
  data: { label: s.label, status: 'healthy', cpu: 12, error: 0.01 },
  position: { x: (i % 3) * 250, y: Math.floor(i / 3) * 150 },
  className: 'glass rounded-lg p-4 w-48 text-xs font-mono',
  style: { background: 'rgba(17, 24, 39, 0.8)', color: '#fff', border: '1px solid rgba(255,255,255,0.1)' }
}));

const INITIAL_EDGES = [
  { id: 'e1-2', source: 'web-gateway', target: 'api-gateway', animated: true },
  { id: 'e2-3', source: 'api-gateway', target: 'order-service', animated: true },
  { id: 'e3-4', source: 'order-service', target: 'postgres-primary', animated: true },
];

// --- Components ---

const MetricCard = ({ icon: Icon, label, value, color }: any) => (
  <div className="glass p-4 rounded-xl flex items-center space-x-4">
    <div className={`p-3 rounded-lg ${color} bg-opacity-20`}>
      <Icon className={`w-6 h-6 ${color.replace('bg-', 'text-')}`} />
    </div>
    <div>
      <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold text-white">{value}</p>
    </div>
  </div>
);

const ThoughtStream = ({ thoughts }: { thoughts: string[] }) => (
  <div className="glass h-full rounded-2xl flex flex-col overflow-hidden">
    <div className="p-4 border-b border-white/10 flex items-center space-x-2 bg-white/5">
      <Brain className="w-5 h-5 text-primary" />
      <h2 className="font-bold text-sm tracking-tight">AGENT REASONING FEED</h2>
    </div>
    <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-[11px]">
      <AnimatePresence>
        {thoughts.map((t, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex space-x-2 group"
          >
            <span className="text-primary/50">[{new Date().toLocaleTimeString()}]</span>
            <span className="text-gray-300 group-hover:text-white transition-colors">{t}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  </div>
);

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(INITIAL_NODES as any);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES);
  const [thoughts, setThoughts] = useState<string[]>([
    "Dashboard connected to core...",
    "Waiting for GPU training data stream...",
  ]);
  const [metrics, setMetrics] = useState({ health: 100, mttr: 0, reward: 0 });

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.episode !== undefined) {
          setMetrics({
            health: 100 + (data.total_reward * 10),
            mttr: data.mttr,
            reward: data.total_reward
          });
          
          setThoughts(prev => [
            ...prev, 
            `Episode ${data.episode} Complete: Reward=${data.total_reward.toFixed(2)} MTTR=${data.mttr}`
          ].slice(-50));
        }
      } catch (e) {
        // Handle raw thought lines if they are not JSON
        setThoughts(prev => [...prev, event.data].slice(-50));
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="w-screen h-screen bg-[#030712] text-white flex flex-col overflow-hidden p-6 gap-6">
      {/* Header */}
      <header className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center shadow-lg shadow-primary/20">
            <ShieldCheck className="w-8 h-8 text-black" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tighter italic">SENTINEL <span className="text-primary">CORE</span></h1>
            <p className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">Multi-Agent SRE Reasoning Engine</p>
          </div>
        </div>
        <div className="flex space-x-2">
          <div className="glass px-4 py-2 rounded-lg flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-mono">NEURAL_LINK: ACTIVE</span>
          </div>
          <div className="glass px-4 py-2 rounded-lg flex items-center space-x-2">
            <span className="text-xs font-mono text-gray-400">FPS: 60.0</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
        
        {/* Left: Metrics & Controls */}
        <div className="col-span-3 flex flex-col gap-6">
          <MetricCard icon={Activity} label="SLA Health" value={`${metrics.health.toFixed(1)}%`} color="bg-emerald-500" />
          <MetricCard icon={Clock} label="Current MTTR" value={`${metrics.mttr} steps`} color="bg-amber-500" />
          <MetricCard icon={Zap} label="Last Reward" value={metrics.reward.toFixed(2)} color="bg-cyan-500" />
          
          <div className="flex-1 glass rounded-2xl p-6 flex flex-col">
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-bold text-xs uppercase tracking-widest text-gray-400">Agent Status</h3>
              <div className="px-2 py-0.5 rounded bg-primary/10 border border-primary/20 text-[10px] text-primary">REASONING</div>
            </div>
            <div className="space-y-4">
              {['HOLMES', 'FORGE', 'ORACLE', 'ARGUS'].map(agent => (
                <div key={agent} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center font-bold text-[10px]">{agent[0]}</div>
                    <span className="text-sm font-medium">{agent}</span>
                  </div>
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center: Service Map */}
        <div className="col-span-6 glass rounded-3xl overflow-hidden relative border border-white/5 shadow-2xl">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            theme="dark"
          >
            <Background color="#111" gap={20} />
            <Controls className="!bg-surface !border-border" />
            <Panel position="top-right" className="glass p-2 rounded-lg">
              <div className="flex items-center space-x-2 text-[10px] font-mono">
                <Globe className="w-3 h-3 text-primary" />
                <span>NEXA_STACK TOPOLOGY</span>
              </div>
            </Panel>
          </ReactFlow>
        </div>

        {/* Right: Thought Stream */}
        <div className="col-span-3">
          <ThoughtStream thoughts={thoughts} />
        </div>
      </div>

      {/* Footer */}
      <footer className="h-8 flex justify-between items-center text-[10px] text-gray-600 font-mono">
        <div>PROD_VERSION: 1.2.0-STABLE</div>
        <div className="flex space-x-4">
          <span>GPU_LOAD: 42%</span>
          <span>MEM: 3.2GB / 4.0GB</span>
        </div>
      </footer>
    </div>
  );
}
