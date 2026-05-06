'use client'

import { useEffect, useState } from 'react'

import { createClient } from '@/lib/supabase/client'

type ProjectSettingsModalProps = {
  project: {
    id: string
    title: string
    created_at: string
  }
  onClose: () => void
  onUpdated: (project: { id: string; title: string }) => void
  onDeleted: (projectId: string) => void
}

export default function ProjectSettingsModal({
  project,
  onClose,
  onUpdated,
  onDeleted,
}: ProjectSettingsModalProps) {
  const [title, setTitle] = useState(project.title)
  const [isSaving, setIsSaving] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    setTitle(project.title)
  }, [project.title])

  const handleSave = async () => {
    if (!title.trim() || isSaving || isDeleting) {
      return
    }

    setIsSaving(true)

    try {
      const response = await fetch(`/api/projects/${project.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim() }),
      })

      const result = (await response.json().catch(() => null)) as
        | { success?: boolean; error?: string; project?: { id: string; title: string } }
        | null

      if (!response.ok) {
        throw new Error(result?.error || '프로젝트 수정에 실패했습니다.')
      }

      if (result?.project) {
        onUpdated(result.project)
      }

      onClose()
    } catch (error) {
      alert(
        error instanceof Error ? error.message : '프로젝트 수정에 실패했습니다.'
      )
    } finally {
      setIsSaving(false)
    }
  }

  const handleDelete = async () => {
    if (isSaving || isDeleting) {
      return
    }

    const confirmed = window.confirm(
      '이 프로젝트를 삭제하면 관련 채팅, 단계 기록, 레퍼런스 이미지, 저장된 파일이 모두 삭제됩니다. 계속할까요?'
    )

    if (!confirmed) {
      return
    }

    setIsDeleting(true)

    try {
      const response = await fetch(`/api/projects/${project.id}`, {
        method: 'DELETE',
      })

      const result = (await response.json().catch(() => null)) as
        | { success?: boolean; error?: string; deletedProjectId?: string }
        | null

      if (!response.ok) {
        throw new Error(result?.error || '프로젝트 삭제에 실패했습니다.')
      }

      onDeleted(result?.deletedProjectId ?? project.id)
      onClose()
    } catch (error) {
      alert(
        error instanceof Error ? error.message : '프로젝트 삭제에 실패했습니다.'
      )
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-[520px] overflow-hidden rounded-[28px] bg-white shadow-2xl">
        <div className="border-b border-gray-100 px-6 py-5">
          <div className="text-lg font-semibold text-neutral-900">프로젝트 수정</div>
          <div className="mt-1 text-sm text-slate-500">프로젝트 이름을 변경하거나 삭제할 수 있습니다.</div>
        </div>

        <div className="space-y-6 px-6 py-6">
          <label className="block space-y-2">
            <span className="text-sm font-medium text-slate-600">프로젝트 이름</span>
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="w-full rounded-2xl border border-gray-200 px-4 py-3 text-sm text-neutral-900 outline-none transition focus:border-blue-300"
              placeholder="프로젝트 이름을 입력하세요"
            />
          </label>

          <div className="rounded-2xl bg-gray-50 px-4 py-3 text-xs leading-5 text-slate-500">
            생성일: {new Date(project.created_at).toLocaleDateString('ko-KR')}
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-gray-100 px-6 py-5">
          <button
            type="button"
            onClick={handleDelete}
            disabled={isSaving || isDeleting}
            className="rounded-full bg-red-50 px-4 py-2 text-sm font-medium text-red-600 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isDeleting ? '삭제 중...' : '프로젝트 삭제'}
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSaving || isDeleting}
              className="rounded-full bg-gray-100 px-4 py-2 text-sm font-medium text-slate-600 transition hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-60"
            >
              취소
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={!title.trim() || isSaving || isDeleting}
              className="rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSaving ? '저장 중...' : '저장'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
