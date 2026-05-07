import Link from 'next/link'
import Image from 'next/image'
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
        <div className="space-y-5">
          <Image
            src="/brand/aidee-logo-blue.svg"
            alt="Aidee"
            width={115}
            height={40}
            unoptimized
            priority
            className="h-10 w-auto"
          />
          <h1 className="text-3xl font-semibold tracking-tight">
            아이디어를 제품디자인 실행으로 연결하세요.
          </h1>
          <p className="text-sm leading-6 text-zinc-600">
            Aidee 베타 테스트에 참여해 주셔서 감사합니다.
            현재 버전에서는 제품 디자인 워크플로우의 핵심 흐름을 먼저 검증하고 있으며, 테스트 결과는 정식 버전 개선에 반영됩니다.
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
            href="https://aidee-studio.com/"
            className="inline-flex h-11 items-center justify-center rounded-xl border border-zinc-200 bg-white px-5 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
          >
            랜딩 페이지로 이동
          </Link>
        </div>
      </div>
    </main>
  )
}
