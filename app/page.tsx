import Link from 'next/link'
import { redirect } from 'next/navigation'

import { createClient } from '@/lib/supabase/server'

export default async function Home() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect('/dashboard')
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-zinc-50 px-6 text-zinc-950">
      <div className="w-full max-w-xl rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm">
        <div className="space-y-3">
          <p className="text-sm font-medium text-sky-700">Aidee</p>
          <h1 className="text-3xl font-semibold tracking-tight">
            제품 디자인 워크플로우를 한 화면에서 이어가세요.
          </h1>
          <p className="text-sm leading-6 text-zinc-600">
            처음 방문한 사용자는 랜딩에서 제품을 이해하고 로그인으로 진입하고,
            이미 로그인된 사용자는 세션을 유지한 채 바로 대시보드로 이동합니다.
          </p>
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            href="/login"
            className="inline-flex h-11 items-center justify-center rounded-xl bg-zinc-950 px-5 text-sm font-medium text-white transition hover:bg-zinc-800"
          >
            시작하기
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex h-11 items-center justify-center rounded-xl border border-zinc-200 bg-white px-5 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
          >
            대시보드 안내 보기
          </Link>
        </div>
      </div>
    </main>
  )
}
