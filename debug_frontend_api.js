// Quick test script to check frontend API calls
const API_URL = 'http://localhost:8000';

const testRequest = {
    symbol: 'QQQ',
    data_source: 'csv', 
    include_indicators: ['rsi', 'macd'],
    rsi_period: 14,
    macd_fast: 12,
    macd_slow: 26,
    macd_signal: 9,
    bb_period: 20,
    bb_std_dev: 2.0,
    sma_periods: [20],
    ema_periods: [12]
};

console.log('Testing frontend API call...');
console.log('Request:', testRequest);

fetch(`${API_URL}/api/v1/analysis/technical-indicators`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(testRequest)
})
.then(response => {
    console.log('Response status:', response.status);
    return response.json();
})
.then(data => {
    console.log('Response data:');
    console.log('- Symbol:', data.symbol);
    console.log('- Indicators count:', data.indicators?.length || 0);
    console.log('- Overall signal:', data.overall_signal);
    
    if (data.indicators && data.indicators.length > 0) {
        data.indicators.forEach((indicator, idx) => {
            console.log(`- Indicator ${idx + 1}:`, indicator.name);
            console.log(`  - Current value:`, indicator.current_value);
            console.log(`  - Signal:`, indicator.signal);
            console.log(`  - Data points:`, indicator.data?.length || 0);
            if (indicator.data && indicator.data.length > 0) {
                console.log(`  - Sample data:`, indicator.data.slice(0, 2));
            }
        });
    }
})
.catch(error => {
    console.error('Error:', error);
});
