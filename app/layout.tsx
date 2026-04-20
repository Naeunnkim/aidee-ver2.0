import type { Metadata } from 'next'

import './globals.css'

export const metadata: Metadata = {
  title: 'Aidee',
  description: '제품 디자인 전문 AI 멀티 에이전트 플랫폼',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body
        className="min-h-full flex flex-col bg-white text-zinc-950"
        style={{
          fontFamily:
            'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }}
      >
        {children}
      </body>
    </html>
  )
}
