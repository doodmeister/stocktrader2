'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Simple test component to verify charts work at all
export default function ChartTest() {
  const testData = [
    { name: 'A', value: 10 },
    { name: 'B', value: 20 },
    { name: 'C', value: 15 },
    { name: 'D', value: 25 },
  ]

  console.log('ChartTest rendering with data:', testData)

  return (
    <div className="p-4 border rounded">
      <h3>Chart Test Component</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={testData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
