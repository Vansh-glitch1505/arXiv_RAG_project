import { useState } from 'react';

export default function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError('');
    setAnswer('');
    setSources([]);

    try {
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message || 'Failed to connect to the backend server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>arXiv Research Assistant</h1>

      <form onSubmit={handleAsk} className="search-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about recent NLP/AI research..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>

      {loading && <p className="status-msg">Loading...</p>}
      {error && <p className="error-msg">{error}</p>}

      {answer && (
        <div className="results">
          <h2>Answer</h2>
          <p className="answer-text">{answer}</p>

          {sources.length > 0 && (
            <div className="sources">
              <h3>Sources</h3>
              <ul>
                {sources.map((source, index) => (
                  <li key={index}>{source}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
