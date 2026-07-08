import React from 'react';
import { HashRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { LayoutDashboard, Route as RouteIcon, Layers, Columns, Cpu, Network } from 'lucide-react';

import Overview from './pages/Overview';
import LaneDetection from './pages/LaneDetection';
import FeatureMatching from './pages/FeatureMatching';
import Panorama from './pages/Panorama';
import Classification from './pages/Classification';

const App: React.FC = () => {
  return (
    <HashRouter>
      <div className="flex min-h-screen bg-dark-950 font-sans text-gray-200">
        
        {/* Sidebar Nav */}
        <aside className="w-64 bg-dark-900 border-r border-gray-800/80 flex flex-col fixed inset-y-0 left-0 z-20">
          {/* Logo Header */}
          <div className="p-6 border-b border-gray-800/80 flex items-center gap-2.5">
            <div className="p-2 bg-brand-500/10 rounded-xl text-brand-500">
              <Network className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">PerCV</h2>
              <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">Vision Benchmark</p>
            </div>
          </div>

          {/* Nav Links */}
          <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
            <NavLink
              to="/overview"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/15'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/50'
                }`
              }
            >
              <LayoutDashboard className="w-5 h-5" /> Overview
            </NavLink>

            <NavLink
              to="/lanes"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/15'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/50'
                }`
              }
            >
              <RouteIcon className="w-5 h-5" /> Lane Detection
            </NavLink>

            <NavLink
              to="/matching"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/15'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/50'
                }`
              }
            >
              <Layers className="w-5 h-5" /> Feature Matching
            </NavLink>

            <NavLink
              to="/panorama"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/15'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/50'
                }`
              }
            >
              <Columns className="w-5 h-5" /> Panorama Stitching
            </NavLink>

            <NavLink
              to="/classification"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/15'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/50'
                }`
              }
            >
              <Cpu className="w-5 h-5" /> CNN Classifier
            </NavLink>
          </nav>

          {/* Footer Status */}
          <div className="p-4 border-t border-gray-800/80 bg-dark-950/20 text-center">
            <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-green-400 bg-green-500/5 px-2.5 py-1 rounded-full border border-green-500/10">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-ping" />
              API Connected
            </span>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 pl-64 min-h-screen">
          <div className="max-w-7xl mx-auto p-8 lg:p-12">
            <Routes>
              <Route path="/overview" element={<Overview />} />
              <Route path="/lanes" element={<LaneDetection />} />
              <Route path="/matching" element={<FeatureMatching />} />
              <Route path="/panorama" element={<Panorama />} />
              <Route path="/classification" element={<Classification />} />
              <Route path="*" element={<Navigate to="/overview" replace />} />
            </Routes>
          </div>
        </main>

      </div>
    </HashRouter>
  );
};

export default App;
