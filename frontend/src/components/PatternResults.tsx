import type { PatternDetectionResponse } from '@/lib/api';
import { useEffect, useState } from 'react';
import PatternChart from './PatternChart';
import { loadOHLCVFromAPI, OHLCV } from '@/lib/ohlcv';

interface PatternResultsProps {
    results: PatternDetectionResponse;
    csvFilePath?: string; // Pass the CSV file path for the current symbol
}

export default function PatternResults({ results, csvFilePath }: PatternResultsProps) {
    const [ohlcv, setOhlcv] = useState<OHLCV[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!csvFilePath) return;

        const filename = csvFilePath.split('/').pop();
        if (!filename) {
            setError("Invalid CSV file path provided.");
            return;
        }

        setLoading(true);
        setError(null);
        loadOHLCVFromAPI(filename)
            .then(setOhlcv)
            .catch(e => {
                const msg = e instanceof Error ? e.message : String(e);
                setError('Failed to load OHLCV data: ' + msg);
                // Log for debugging
                console.error('OHLCV load error:', msg, 'File:', filename);
            })
            .finally(() => setLoading(false));
    }, [csvFilePath]);

    if (!results) { // Defensive check for results
        return null;
    }

    if (results.patterns.length === 0) {
        return (
            <div className="trading-card">
                <h2 className="text-lg font-semibold mb-4">Pattern Detection Results for {results.symbol}</h2>
                <p className="text-sm text-muted-foreground">No patterns found matching the criteria.</p>
            </div>
        )
    }

    return (
        <div className="trading-card">
            <h2 className="text-lg font-semibold mb-4">Pattern Detection Results for {results.symbol}</h2>
            <p className="text-sm text-muted-foreground mb-4">
                {results.pattern_summary
                    ? `Found ${results.pattern_summary.total_patterns} pattern instances.`
                    : 'No pattern summary available.'}
            </p>
            {/* Chart overlay for detected patterns */}
            {loading && <div>Loading chart...</div>}
            {error && <div className="text-red-500">{error}</div>}
            {ohlcv && ohlcv.length > 0 && !error && (
                <div className="mb-6">
                    <PatternChart ohlcv={ohlcv} patterns={results.patterns} />
                </div>
            )}
            <div className="overflow-x-auto max-h-96">
                <table className="min-w-full divide-y divide-border">
                    <thead className="bg-muted/50 sticky top-0">
                        <tr>
                            <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Start Index</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Pattern Name</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Type</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Confidence</th>
                        </tr>
                    </thead>
                    <tbody className="bg-background divide-y divide-border">
                        {results.patterns.map((pattern, index) => (
                            <tr key={index}>
                                <td className="px-4 py-2 whitespace-nowrap text-sm">{pattern.start_index}</td>
                                <td className="px-4 py-2 whitespace-nowrap text-sm font-medium">{pattern.name}</td>
                                <td className="px-4 py-2 whitespace-nowrap text-sm">{pattern.signal}</td>
                                <td className="px-4 py-2 whitespace-nowrap text-sm">{(pattern.confidence * 100).toFixed(1)}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
