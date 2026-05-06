'use client'

import Link from 'next/link'
import { useCallback, useEffect, useState } from 'react'

import ProjectModal from '@/components/dashboard/project-modal'
import ProjectSettingsModal from '@/components/dashboard/project-settings-modal'
import { extractGeneratedImagesBlock } from '@/lib/image-generation'
import { saveGeneratedProjectThumbnail } from '@/lib/project-thumbnail'
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
  requirements?: {
    thumbnail_image_url?: unknown
    thumbnail_image_source?: unknown
    thumbnail_image_prompt?: unknown
  } | null
}

type MessageRecord = {
  content: string
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

function getProjectThumbnail(project: Project) {
  const thumbnailUrl = project.requirements?.thumbnail_image_url

  return typeof thumbnailUrl === 'string' && thumbnailUrl.trim()
    ? thumbnailUrl
    : null
}

export function ProjectList({ user }: ProjectListProps) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)

  const loadProjects = useCallback(async () => {
    const supabase = createClient()
    const {
      data: { user: authUser },
    } = await supabase.auth.getUser()

    if (!authUser) {
      setProjects([])
      setLoading(false)
      return
    }

    const { data, error } = await supabase
      .from('projects')
      .select('id, title, created_at, requirements')
      .eq('user_id', authUser.id)
      .order('created_at', { ascending: false })

    if (error) {
      setProjects([])
      setLoading(false)
      return
    }

    const nextProjects = (data ?? []) as Project[]
    setProjects(nextProjects)
    setLoading(false)

    return nextProjects
  }, [])

  useEffect(() => {
    let isMounted = true

    async function backfillProjectThumbnail(project: Project) {
      if (getProjectThumbnail(project)) {
        return
      }

      const supabase = createClient()
      const { data: messages, error } = await supabase
        .from('messages')
        .select('content')
        .eq('project_id', project.id)
        .eq('role', 'assistant')
        .order('seq_order', { ascending: false })
        .limit(30)

      if (error) {
        console.error('[dashboard thumbnail] failed to fetch messages:', error)
        return
      }

      const imageBlock = ((messages ?? []) as MessageRecord[])
        .map((message) => extractGeneratedImagesBlock(message.content).imageBlock)
        .find((block) => block?.images.length)

      if (!imageBlock) {
        return
      }

      try {
        const thumbnailUrl = await saveGeneratedProjectThumbnail({
          supabase,
          projectId: project.id,
          imageBlock,
          overwrite: false,
        })

        if (!thumbnailUrl || !isMounted) {
          return
        }

        setProjects((prev) =>
          prev.map((item) =>
            item.id === project.id
              ? {
                  ...item,
                  requirements: {
                    ...(item.requirements ?? {}),
                    thumbnail_image_url: thumbnailUrl,
                    thumbnail_image_source: 'generated',
                    thumbnail_image_prompt: imageBlock.prompt,
                  },
                }
              : item
          )
        )
      } catch (thumbnailError) {
        console.error(
          '[dashboard thumbnail] failed to backfill generated thumbnail:',
          thumbnailError
        )
      }
    }

    void loadProjects().then((nextProjects) => {
      if (!isMounted || !nextProjects) {
        return
      }

      nextProjects
        .filter((project) => !getProjectThumbnail(project))
        .forEach((project) => {
          void backfillProjectThumbnail(project)
        })
    })

    return () => {
      isMounted = false
    }
  }, [loadProjects])

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
                <div
                  key={project.id}
                  className="group relative flex h-64 flex-col overflow-hidden rounded-[24px] border border-gray-100 bg-white shadow-sm transition-all hover:shadow-md"
                >
                  <div className="absolute right-3 top-3 z-10 flex gap-2 opacity-100 transition md:opacity-0 md:group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => setSelectedProject(project)}
                      className="rounded-full bg-white/95 px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-blue-50 hover:text-blue-700"
                    >
                      수정
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        const confirmed = window.confirm(
                          '이 프로젝트를 삭제할까요? 관련 데이터가 모두 삭제됩니다.'
                        )

                        if (!confirmed) {
                          return
                        }

                        const response = await fetch(`/api/projects/${project.id}`, {
                          method: 'DELETE',
                        })

                        if (!response.ok) {
                          const result = (await response.json().catch(() => null)) as
                            | { error?: string }
                            | null
                          alert(result?.error || '프로젝트 삭제에 실패했습니다.')
                          return
                        }

                        setProjects((prev) =>
                          prev.filter((item) => item.id !== project.id)
                        )
                      }}
                      className="rounded-full bg-white/95 px-3 py-1.5 text-xs font-semibold text-red-600 shadow-sm outline outline-1 outline-red-100 transition hover:bg-red-50"
                    >
                      삭제
                    </button>
                  </div>

                  <Link href={`/project/${project.id}`} className="flex h-full flex-col">
                    <div className="relative flex h-[140px] items-center justify-center overflow-hidden bg-slate-50">
                      {getProjectThumbnail(project) ? (
                        <img
                          src={getProjectThumbnail(project) ?? ''}
                          alt={`${project.title} 썸네일`}
                          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                        />
                      ) : (
                        <div className="h-full w-full bg-gradient-to-br from-gray-100 to-gray-200" />
                      )}
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
                </div>
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

      {selectedProject ? (
        <ProjectSettingsModal
          project={selectedProject}
          onClose={() => setSelectedProject(null)}
          onUpdated={(updatedProject) => {
            setProjects((prev) =>
              prev.map((project) =>
                project.id === updatedProject.id
                  ? { ...project, title: updatedProject.title }
                  : project
              )
            )
          }}
          onDeleted={(deletedProjectId) => {
            setProjects((prev) =>
              prev.filter((project) => project.id !== deletedProjectId)
            )
          }}
        />
      ) : null}

      {isModalOpen ? <ProjectModal onClose={() => setIsModalOpen(false)} /> : null}
    </div>
  )
}
