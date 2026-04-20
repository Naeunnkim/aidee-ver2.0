import Link from 'next/link'
import { redirect } from 'next/navigation'

import { createClient } from '@/lib/supabase/server'

const categories = ['전체', '인테리어', '생활 용품', '패션 악세서리']

const projects = [
  {
    title: 'Foldable 벽조명',
    status: '스타일 구체화',
    statusClass: 'bg-amber-100 text-amber-600',
    updatedAt: '마지막 수정: 1일 전',
  },
  {
    title: '컬러 유리 캔들 홀더',
    status: '리서치 요약',
    statusClass: 'bg-orange-100 text-orange-600',
    updatedAt: '마지막 수정: 3일 전',
  },
  {
    title: '리서치 요약 정리 완료',
    status: 'CMF 전략 반영',
    statusClass: 'bg-violet-100 text-violet-600',
    updatedAt: '마지막 수정: 12일 전',
  },
  {
    title: 'Warm Wood 무드등',
    status: '아이디어 구체화',
    statusClass: 'bg-rose-100 text-rose-600',
    updatedAt: '마지막 수정: 18일 전',
  },
  {
    title: 'Float Light 천장 고정형 조명 디자인',
    updatedAt: '2025.08.01',
  },
  {
    title: 'ColorCatcher 반응형 무드 오브제',
    updatedAt: '2025.08.01',
  },
  {
    title: 'Tento Shelf 모듈형 미니 선반',
    updatedAt: '2025.08.01',
  },
]

export default async function DashboardPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect('/login')
  }

  const displayName =
    user.user_metadata.full_name ??
    user.user_metadata.name ??
    user.email?.split('@')[0] ??
    '사용자'

  return (
    <main className="min-h-screen overflow-hidden bg-white text-zinc-950">
      <div className="relative mx-auto min-h-screen max-w-[1440px]">
        <header className="border-b border-zinc-200 bg-white px-6 py-3">
          <div className="flex h-7 items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-7 w-20 rounded-full bg-gradient-to-r from-sky-500 to-blue-700" />
              <div className="text-sm font-medium text-zinc-500">
                {displayName}님
              </div>
            </div>

            <div className="inline-flex items-center gap-2 rounded-full bg-gradient-to-bl from-blue-600/0 to-blue-600/40 px-4 py-1 text-sm font-medium text-white">
              <div className="flex h-4 w-4 items-center justify-center rounded-full border border-white/80 text-[10px]">
                ✓
              </div>
              <div>Basic</div>
            </div>
          </div>
        </header>

        <section className="px-[70px] py-16">
          <div className="flex flex-col items-center gap-4">
            <div className="flex w-full flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap items-center gap-1.5">
                {categories.map((category, index) => (
                  <button
                    key={category}
                    type="button"
                    className={
                      index === 0
                        ? 'rounded-full border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-semibold text-sky-700'
                        : 'rounded-full bg-zinc-100 px-3 py-1.5 text-xs font-semibold text-zinc-600'
                    }
                  >
                    {category}
                  </button>
                ))}
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <div className="flex h-11 w-full min-w-0 items-center justify-between rounded-full border border-zinc-300 bg-white px-5 pr-4 text-base text-zinc-400 shadow-[0px_2.16px_4.32px_-1.08px_rgba(0,0,0,0.10)] sm:w-96">
                  <span>찾고있는 프로젝트를 검색해주세요.</span>
                  <span className="flex h-6 w-6 items-center justify-center rounded-full border border-zinc-300 text-xs text-zinc-400">
                    ⌕
                  </span>
                </div>

                <button
                  type="button"
                  className="inline-flex items-center justify-center rounded-full bg-zinc-950 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-zinc-800"
                >
                  프로젝트 생성
                </button>
              </div>
            </div>

            <div className="flex w-full flex-wrap items-start gap-2.5">
              {projects.map((project) => (
                <article
                  key={project.title}
                  className="w-full sm:w-[calc(50%-0.3125rem)] lg:w-[calc(25%-0.46875rem)]"
                >
                  <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
                    <div className="h-32 bg-zinc-200" />
                    <div className="flex flex-col gap-2.5 p-2.5">
                      <div className="flex flex-col gap-1.5">
                        <h2 className="text-sm font-semibold leading-5 text-neutral-900">
                          {project.title}
                        </h2>

                        {project.status ? (
                          <div className="inline-flex items-center gap-1.5">
                            <span
                              className={`rounded-[19px] px-1.5 py-0.5 text-xs font-medium leading-4 ${project.statusClass}`}
                            >
                              {project.status}
                            </span>
                            <span className="text-[10px] leading-4 text-zinc-400">
                              {project.updatedAt}
                            </span>
                          </div>
                        ) : (
                          <div className="text-[10px] leading-4 text-zinc-400">
                            {project.updatedAt}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </article>
              ))}
            </div>

            <div className="flex w-full items-center justify-between rounded-2xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-3 text-sm text-zinc-500">
              <span>{user.email}</span>
              <div className="flex items-center gap-3">
                <span className="hidden text-zinc-400 md:inline">{user.id}</span>
                <Link href="/login" className="font-medium text-zinc-700">
                  로그인 화면
                </Link>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
