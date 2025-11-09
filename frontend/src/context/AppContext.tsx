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
      try {
        const parsed: VoiceTest[] = JSON.parse(stored);
        // Trim any heavy/base64 fields from stored data (migration for prior versions)
        const trimmed = parsed.map(trimTestForStorage);
        setTests(trimmed);
        // Persist back the trimmed version to free up quota
        localStorage.setItem('voicecare_tests', JSON.stringify(trimmed));
      } catch (e) {
        console.warn('Failed to parse cached tests, clearing corrupt storage', e);
        localStorage.removeItem('voicecare_tests');
      }
    }
  }, []);

  const addTest = (test: VoiceTest) => {
    const updated = [test, ...tests];
    setTests(updated);

    // Store a lightweight copy without large base64 images to avoid quota issues
    let persistable = updated.map(trimTestForStorage);
    try {
      localStorage.setItem('voicecare_tests', JSON.stringify(persistable));
    } catch (err) {
      console.warn('localStorage quota exceeded, attempting to reduce payload...', err);
      // Gradually drop oldest entries until it fits
      while (persistable.length > 0) {
        try {
          persistable = persistable.slice(0, -1);
          localStorage.setItem('voicecare_tests', JSON.stringify(persistable));
          console.info('Reduced stored history to fit quota. Stored count:', persistable.length);
          break;
        } catch (_) {
          // keep trimming
        }
      }
    }
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

// Helpers
function trimTestForStorage(test: VoiceTest): VoiceTest {
  const t: any = { ...test };
  if (t.shap_analysis) {
    const { visualization_base64, heatmap_base64, ...restShap } = t.shap_analysis as any;
    t.shap_analysis = { ...restShap };
  }
  return t as VoiceTest;
}

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
};
