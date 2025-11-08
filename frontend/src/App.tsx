import { AppProvider, useApp } from './context/AppContext';
import { SplashPage } from './pages/SplashPage';
import { HomePage } from './pages/HomePage';
import { RecordPage } from './pages/RecordPage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { ReportPage } from './pages/ReportPage';
import { ChatWidget } from './components/ChatWidget';

function AppContent() {
  const { currentPage } = useApp();

  const renderPage = () => {
    switch (currentPage) {
      case 'splash':
        return <SplashPage />;
      case 'home':
        return <HomePage />;
      case 'record':
        return <RecordPage />;
      case 'results':
        return <ResultsPage />;
      case 'history':
        return <HistoryPage />;
      case 'report':
        return <ReportPage />;
      default:
        return <HomePage />;
    }
  };

  return <>
    {renderPage()}
    {/* Floating Chatbot */}
    <ChatWidget apiBase={'http://localhost:5000'} />
  </>;
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
