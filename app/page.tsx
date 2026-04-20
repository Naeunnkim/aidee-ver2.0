import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-zinc-50 px-6 text-zinc-950">
      <div className="w-full max-w-xl rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm">
        <div className="space-y-3">
          <p className="text-sm font-medium text-sky-700">Aidee</p>
          <h1 className="text-3xl font-semibold tracking-tight">
            로그인 페이지 작업을 시작했습니다.
          </h1>
          <p className="text-sm leading-6 text-zinc-600">
            현재 로그인 화면 뼈대는 별도 경로에 준비되어 있습니다. Figma에서
            복사한 Tailwind UI는 그 구조에 맞춰 바로 붙여넣으면 됩니다.
          </p>
        </div>

        <div className="mt-6">
          <Link
            href="/login"
            className="inline-flex h-11 items-center justify-center rounded-xl bg-zinc-950 px-5 text-sm font-medium text-white transition hover:bg-zinc-800"
          >
            로그인 페이지 보기
          </Link>
        </div>
      </div>
    </main>
  )
}
