import { AppProvider, useApp } from './context/AppContext';
import { SplashPage } from './pages/SplashPage';
import { HomePage } from './pages/HomePage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { ReportPage } from './pages/ReportPage';

function AppContent() {
  const { currentPage } = useApp();

  const renderPage = () => {
    switch (currentPage) {
      case 'splash':
        return <SplashPage />;
      case 'home':
        return <HomePage />;
      case 'record':
        return <UploadPage />;
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

  return <>{renderPage()}</>;
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
