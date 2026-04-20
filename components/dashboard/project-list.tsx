'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'

import ProjectModal from '@/components/dashboard/project-modal'
import { createClient } from '@/lib/supabase/client'

type DashboardUser = {
  displayName: string
  email: string
  id: string
}

type Project = {
  id: string
  title: string
  created_at: string
}

type ProjectListProps = {
  user: DashboardUser
}

const categories = ['전체', '인테리어', '생활 용품']

function formatProjectDate(value: string) {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return date.toLocaleDateString('ko-KR')
}

export function ProjectList({ user }: ProjectListProps) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    let isMounted = true

    async function fetchProjects() {
      const supabase = createClient()
      const {
        data: { user: authUser },
      } = await supabase.auth.getUser()

      if (!authUser) {
        if (isMounted) {
          setProjects([])
          setLoading(false)
        }

        return
      }

      const { data, error } = await supabase
        .from('projects')
        .select('id, title, created_at')
        .eq('user_id', authUser.id)
        .order('created_at', { ascending: false })

      if (error) {
        // The new project currently has no `projects` table.
        if (isMounted) {
          setProjects([])
          setLoading(false)
        }

        return
      }

      if (isMounted) {
        setProjects((data ?? []) as Project[])
        setLoading(false)
      }
    }

    fetchProjects()

    return () => {
      isMounted = false
    }
  }, [])

  const filteredProjects = projects.filter((project) =>
    project.title.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="w-full border-b border-gray-200 bg-white px-6 py-3">
        <div className="mx-auto flex w-full max-w-[1300px] items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-7 w-20 rounded-full bg-gradient-to-r from-sky-500 to-blue-700" />
            <div className="text-sm font-medium text-zinc-500">
              {user.displayName}님
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="rounded-full bg-gradient-to-bl from-[#8BEAFF] to-[#4D95FF] px-4 py-1">
              <div className="flex items-center gap-2">
                <span className="text-xs text-white">✦</span>
                <span className="text-sm font-medium text-white">Basic</span>
              </div>
            </div>
            <form action="/auth/logout" method="post">
              <button
                type="submit"
                className="inline-flex h-9 items-center justify-center rounded-full border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
              >
                로그아웃
              </button>
            </form>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1300px] px-6 pt-20 pb-12">
        <div className="space-y-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap gap-2">
              {categories.map((category, index) => (
                <button
                  key={category}
                  type="button"
                  className={
                    index === 0
                      ? 'rounded-full bg-indigo-50 px-4 py-2 text-sm font-medium text-blue-600'
                      : 'rounded-full border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700'
                  }
                >
                  {category}
                </button>
              ))}
            </div>

            <div className="flex w-full items-center gap-3 md:w-auto">
              <div className="flex h-11 flex-1 items-center justify-between rounded-full border border-gray-300 bg-white px-5 shadow-sm md:w-80">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                  placeholder="찾고있는 프로젝트를 검색해주세요."
                  className="w-full bg-transparent text-sm outline-none placeholder:text-zinc-400"
                />
                <span className="text-sm text-zinc-400">⌕</span>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(true)}
                className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700"
              >
                프로젝트 생성
              </button>
            </div>
          </div>

          {loading ? (
            <div className="rounded-[24px] border border-dashed border-zinc-200 bg-white px-6 py-12 text-center text-sm text-zinc-500">
              프로젝트를 불러오는 중...
            </div>
          ) : filteredProjects.length > 0 ? (
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              {filteredProjects.map((project) => (
                <Link
                  key={project.id}
                  href={`/project/${project.id}`}
                  className="group flex h-64 flex-col overflow-hidden rounded-[24px] border border-gray-100 bg-white shadow-sm transition-all hover:shadow-md"
                >
                  <div className="relative flex h-[140px] items-center justify-center overflow-hidden bg-slate-50">
                    <div className="h-full w-full bg-gradient-to-br from-gray-100 to-gray-200" />
                  </div>

                  <div className="flex flex-col gap-2 p-5">
                    <h3 className="truncate text-[15px] leading-tight font-bold text-neutral-900">
                      {project.title}
                    </h3>

                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-amber-100 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-500">
                        기획 단계
                      </span>
                      <span className="text-[10px] font-normal text-zinc-400">
                        {formatProjectDate(project.created_at)}
                      </span>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="rounded-[24px] border border-dashed border-zinc-200 bg-white px-6 py-12 text-center">
              <div className="space-y-2">
                <p className="text-base font-semibold text-zinc-900">
                  아직 프로젝트가 없습니다.
                </p>
                <p className="text-sm leading-6 text-zinc-500">
                  현재 Supabase에 `projects` 테이블이 없거나, 생성된 프로젝트가
                  없는 상태입니다.
                </p>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between rounded-2xl border border-dashed border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-500">
            <span>{user.email}</span>
            <div className="flex items-center gap-3">
              <span className="hidden text-zinc-400 md:inline">{user.id}</span>
              <Link href="/" className="font-medium text-zinc-700">
                랜딩 페이지
              </Link>
            </div>
          </div>
        </div>
      </main>

      {isModalOpen ? <ProjectModal onClose={() => setIsModalOpen(false)} /> : null}
    </div>
  )
}
