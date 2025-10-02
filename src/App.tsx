import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MessageSquare, Activity, TrendingUp, Clock, FileText, BarChart3, Search, Database, Zap } from 'lucide-react';

interface Query {
  id: number;
  question: string;
  response: string;
  success: boolean;
  created_at: string;
}

interface Stats {
  total_queries: number;
  successful_queries: number;
  success_rate: number;
}

const API_URL = process.env.REACT_APP_API_URL || '';

// Helper functions for table display
const getColumnName = (question: string, index: number, totalCols: number) => {
  const q = question.toLowerCase();
  
  if (q.includes('user')) {
    const userCols = ['ID', 'Name', 'Email', 'Phone', 'Address', 'City', 'Country', 'Created', 'Updated'];
    return userCols[index] || `Col ${index + 1}`;
  }
  
  if (q.includes('product')) {
    const productCols = ['ID', 'Name', 'Description', 'Price', 'Category', 'Stock', 'Active', 'Created', 'Updated'];
    return productCols[index] || `Col ${index + 1}`;
  }
  
  if (q.includes('order')) {
    const orderCols = ['ID', 'User ID', 'Product ID', 'Quantity', 'Unit Price', 'Total', 'Status', 'Order Date', 'Shipped', 'Delivered'];
    return orderCols[index] || `Col ${index + 1}`;
  }
  
  if (q.includes('count') || q.includes('total') || q.includes('sum')) {
    return totalCols === 2 ? (index === 0 ? 'Category' : 'Count') : `Col ${index + 1}`;
  }
  
  return `Column ${index + 1}`;
};

const formatCellValue = (value: any) => {
  if (value === null || value === undefined) return '-';
  
  // Format dates
  if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}T/)) {
    return new Date(value).toLocaleDateString();
  }
  
  // Format numbers
  if (typeof value === 'number') {
    if (value > 1000 && value.toString().includes('.')) {
      return `$${value.toLocaleString()}`;
    }
    return value.toLocaleString();
  }
  
  // Truncate long strings
  if (typeof value === 'string' && value.length > 50) {
    return value.substring(0, 47) + '...';
  }
  
  return value.toString();
};

function App() {
  const [queries, setQueries] = useState<Query[]>([]);
  const [stats, setStats] = useState<Stats>({ total_queries: 0, successful_queries: 0, success_rate: 0 });
  const [testQuestion, setTestQuestion] = useState('');
  const [testResult, setTestResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [charts, setCharts] = useState<any[]>([]);

  useEffect(() => {
    fetchQueries();
    fetchStats();
  }, []);

  const fetchQueries = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/queries`);
      setQueries(response.data);
    } catch (error) {
      console.error('Error fetching queries:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const testQuery = async () => {
    if (!testQuestion.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        question: testQuestion
      });
      setTestResult(response.data);
      fetchQueries();
      fetchStats();
    } catch (error) {
      console.error('Error testing query:', error);
      setTestResult({ error: 'Failed to process query' });
    }
    setLoading(false);
  };

  const exportCSV = async () => {
    if (!testResult || !testResult.raw_data || testResult.raw_data.length === 0) {
      alert('No data to export');
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/api/export-csv`, {
        raw_data: testResult.raw_data,
        question: testResult.question
      });
      
      const csvBlob = new Blob([response.data.csv_content], { type: 'text/csv' });
      const url = URL.createObjectURL(csvBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = response.data.filename;
      link.click();
    } catch (error) {
      alert('Failed to export CSV');
    }
  };

  const generateChart = async () => {
    if (!testResult || !testResult.raw_data || testResult.raw_data.length === 0) {
      alert('No data for chart');
      return;
    }

    try {
      console.log('Generating charts for data:', testResult.raw_data);
      const response = await axios.post(`${API_URL}/api/generate-chart`, {
        raw_data: testResult.raw_data,
        question: testResult.question
      });
      
      console.log('Chart response:', response.data);
      setCharts(response.data.charts || []);
    } catch (error) {
      console.error('Chart generation error:', error);
      alert('Failed to generate charts');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-3 rounded-xl">
              <Database className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Slack AI Data Bot
              </h1>
              <p className="text-gray-500 text-sm">Intelligent data querying dashboard</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 mb-1">Total Queries</p>
                <p className="text-3xl font-bold text-gray-900">{stats.total_queries}</p>
              </div>
              <div className="bg-blue-100 p-3 rounded-xl">
                <MessageSquare className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 mb-1">Successful</p>
                <p className="text-3xl font-bold text-green-600">{stats.successful_queries}</p>
              </div>
              <div className="bg-green-100 p-3 rounded-xl">
                <Activity className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 mb-1">Success Rate</p>
                <p className="text-3xl font-bold text-purple-600">{stats.success_rate.toFixed(1)}%</p>
              </div>
              <div className="bg-purple-100 p-3 rounded-xl">
                <TrendingUp className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Test Query Section */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 mb-8">
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-lg">
                <Search className="h-5 w-5 text-white" />
              </div>
              <h2 className="text-xl font-bold text-gray-900">Query Tester</h2>
            </div>
          </div>
          <div className="p-6">
            <div className="flex gap-4 mb-6">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={testQuestion}
                  onChange={(e) => setTestQuestion(e.target.value)}
                  placeholder="Ask a question about your data... (e.g., 'Show me all users', 'Total orders by category')"
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 placeholder-gray-400"
                  onKeyPress={(e) => e.key === 'Enter' && testQuery()}
                />
              </div>
              <button
                onClick={testQuery}
                disabled={loading}
                className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    <span>Processing...</span>
                  </div>
                ) : (
                  'Run Query'
                )}
              </button>
            </div>
          
            {testResult && (
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-100">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">Query Results</h3>
                  {testResult.raw_data && testResult.raw_data.length > 0 && (
                    <div className="flex gap-3">
                      <button
                        onClick={generateChart}
                        className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all text-sm font-medium shadow-md hover:shadow-lg"
                      >
                        <BarChart3 className="h-4 w-4" />
                        Visualize Data
                      </button>
                      <button
                        onClick={exportCSV}
                        className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors text-sm font-medium"
                      >
                        <FileText className="h-4 w-4" />
                        Export CSV
                      </button>
                    </div>
                  )}
                </div>
                {testResult.error ? (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-700 font-medium">Error:</p>
                    <p className="text-red-600 mt-1">{testResult.error}</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-sm font-medium text-gray-500 mb-2">Generated SQL:</p>
                      <code className="bg-gray-100 px-3 py-2 rounded-lg text-sm text-gray-800 block font-mono">{testResult.sql}</code>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-sm font-medium text-gray-500 mb-2">Response:</p>
                      <p className="text-gray-700 whitespace-pre-wrap mb-4">{testResult.response}</p>
                      
                      {/* Tabular Data Display */}
                      {testResult.raw_data && testResult.raw_data.length > 0 && (
                        <div className="mt-4">
                          <p className="text-sm font-medium text-gray-500 mb-3">Data Table:</p>
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200 border border-gray-200 rounded-lg">
                              <thead className="bg-gray-50">
                                <tr>
                                  {testResult.raw_data[0].map((_: any, index: number) => (
                                    <th key={index} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                      {getColumnName(testResult.question, index, testResult.raw_data[0].length)}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="bg-white divide-y divide-gray-200">
                                {testResult.raw_data.slice(0, 10).map((row: any[], rowIndex: number) => (
                                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                    {row.map((cell: any, cellIndex: number) => (
                                      <td key={cellIndex} className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                                        {formatCellValue(cell)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {testResult.raw_data.length > 10 && (
                              <p className="text-xs text-gray-500 mt-2 text-center">
                                Showing first 10 of {testResult.raw_data.length} rows
                              </p>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                    {charts.length > 0 && (
                      <div className="bg-white rounded-lg p-4 border border-gray-200">
                        <p className="text-sm font-medium text-gray-500 mb-3">Data Visualizations ({charts.length} charts):</p>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          {charts.map((chart: any, index: number) => (
                            <div key={index} className="border border-gray-100 rounded-lg p-4 bg-gray-50">
                              <h5 className="text-sm font-semibold text-gray-700 mb-3">{chart.title || `Chart ${index + 1}`}</h5>
                              <div dangerouslySetInnerHTML={{ __html: chart.html }} />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Recent Queries */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100">
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-r from-green-500 to-blue-500 p-2 rounded-lg">
                  <Activity className="h-5 w-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-gray-900">Query History</h2>
              </div>
              <span className="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-sm font-medium">
                {queries.length} queries
              </span>
            </div>
          </div>
          <div className="overflow-hidden">
            {queries.length === 0 ? (
              <div className="p-12 text-center">
                <div className="bg-gray-100 rounded-full p-4 w-16 h-16 mx-auto mb-4">
                  <MessageSquare className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No queries yet</h3>
                <p className="text-gray-500">Start by mentioning the bot in Slack or test a query above!</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {queries.map((query, index) => (
                  <div key={query.id} className="p-6 hover:bg-gray-50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3 mb-3">
                          <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded-full">
                            #{queries.length - index}
                          </span>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            query.success 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {query.success ? '✓ Success' : '✗ Failed'}
                          </span>
                        </div>
                        <h4 className="font-semibold text-gray-900 mb-2 text-lg">{query.question}</h4>
                        <p className="text-gray-600 mb-3 line-clamp-2">{query.response}</p>
                        <div className="flex items-center text-sm text-gray-500">
                          <Clock className="h-4 w-4 mr-2" />
                          {new Date(query.created_at).toLocaleString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;