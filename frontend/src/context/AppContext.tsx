import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { VoiceTest } from '../types';

interface AppContextType {
  currentPage: string;
  setCurrentPage: (page: string) => void;
  tests: VoiceTest[];
  addTest: (test: VoiceTest) => void;
  clearCache: () => void;
  latestTest: VoiceTest | null;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentPage, setCurrentPage] = useState('splash');
  const [tests, setTests] = useState<VoiceTest[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('voicecare_tests');
    if (stored) {
      setTests(JSON.parse(stored));
    }
  }, []);

  const addTest = (test: VoiceTest) => {
    const updated = [test, ...tests];
    setTests(updated);
    localStorage.setItem('voicecare_tests', JSON.stringify(updated));
  };

  const clearCache = () => {
    setTests([]);
    localStorage.removeItem('voicecare_tests');
    setCurrentPage('home');
  };

  const latestTest = tests.length > 0 ? tests[0] : null;

  return (
    <AppContext.Provider value={{ currentPage, setCurrentPage, tests, addTest, clearCache, latestTest }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
};
