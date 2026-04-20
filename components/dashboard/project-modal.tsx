'use client'

import { useCallback, useState } from 'react'
import { useRouter } from 'next/navigation'

import ProjectForm from '@/components/dashboard/project-form'
import { createClient } from '@/lib/supabase/client'

type ProjectModalProps = {
  onClose: () => void
}

type ReferenceImageRow = {
  project_id: string
  file_name: string
  image_url: string
  analysis_status: 'pending'
  analysis_json: {
    file_path: string
    mime_type: string
    file_size: number
  }
}

export default function ProjectModal({ onClose }: ProjectModalProps) {
  const [step, setStep] = useState(1)
  const [isValid, setIsValid] = useState(false)
  const [formData, setFormData] = useState<Record<string, unknown>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const router = useRouter()

  const handleValidationChange = useCallback((valid: boolean) => {
    setIsValid(valid)
  }, [])

  const handleDataChange = useCallback((data: Record<string, unknown>) => {
    setFormData((prev) => ({ ...prev, ...data }))
  }, [])

  const uploadReferenceImages = async (projectId: string, files: File[]) => {
    const supabase = createClient()
    const uploadedRows: ReferenceImageRow[] = []

    for (const file of files) {
      const fileExt = file.name.split('.').pop() || 'png'
      const filePath = `projects/${projectId}/references/${crypto.randomUUID()}.${fileExt}`

      const { error: uploadError } = await supabase.storage
        .from('project-reference-images')
        .upload(filePath, file, {
          contentType: file.type,
          upsert: false,
        })

      if (uploadError) {
        throw uploadError
      }

      const { data: publicUrlData } = supabase.storage
        .from('project-reference-images')
        .getPublicUrl(filePath)

      uploadedRows.push({
        project_id: projectId,
        file_name: file.name,
        image_url: publicUrlData.publicUrl,
        analysis_status: 'pending',
        analysis_json: {
          file_path: filePath,
          mime_type: file.type,
          file_size: file.size,
        },
      })
    }

    return uploadedRows
  }

  const triggerReferenceImageAnalysis = async (projectId: string) => {
    const response = await fetch('/api/reference-images/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ projectId }),
    })

    const result = await response.json()

    if (!response.ok) {
      throw new Error(result?.error || '레퍼런스 이미지 분석 요청에 실패했습니다.')
    }

    return result
  }

  const handleNext = async () => {
    if (isSubmitting) {
      return
    }

    if (step < 4) {
      setStep(step + 1)
      setIsValid(false)
      return
    }

    try {
      setIsSubmitting(true)
      const supabase = createClient()
      const {
        data: { user },
      } = await supabase.auth.getUser()

      if (!user) {
        throw new Error('로그인이 필요합니다.')
      }

      const safeIdea =
        typeof formData.idea === 'string'
          ? formData.idea
          : typeof formData.title === 'string'
            ? formData.title
            : ''

      const summarizedTitle =
        safeIdea.length > 15
          ? `${safeIdea.slice(0, 15)}...`
          : safeIdea || '새 프로젝트'

      const requirementsForSave = { ...formData }
      delete requirementsForSave.referenceImages

      const { data: project, error: projectError } = await supabase
        .from('projects')
        .insert({
          user_id: user.id,
          title: summarizedTitle,
          requirements: requirementsForSave,
        })
        .select()
        .single()

      if (projectError) {
        throw new Error(`projects insert failed: ${projectError.message}`)
      }

      if (!project) {
        throw new Error('projects insert succeeded without returning a project')
      }

      const referenceImages = Array.isArray(formData.referenceImages)
        ? formData.referenceImages.filter(
            (file: unknown): file is File => file instanceof File
          )
        : []

      let postCreateWarning: string | null = null

      if (referenceImages.length > 0) {
        try {
          const uploadedImageRows = await uploadReferenceImages(
            project.id,
            referenceImages
          )

          const { error: referenceImageInsertError } = await supabase
            .from('project_reference_images')
            .insert(uploadedImageRows)

          if (referenceImageInsertError) {
            throw new Error(
              `project_reference_images insert failed: ${referenceImageInsertError.message}`
            )
          }

          await triggerReferenceImageAnalysis(project.id)
        } catch (error) {
          const message =
            error instanceof Error
              ? error.message
              : 'reference image post-processing failed'
          console.error('[project create] post-create warning:', message)
          postCreateWarning = message
        }
      }

      router.push(`/project/${project.id}?isNew=true`)
      onClose()

      if (postCreateWarning) {
        window.setTimeout(() => {
          alert(
            `프로젝트는 생성되었지만 참고 이미지 처리 중 일부가 실패했습니다.\n${postCreateWarning}`
          )
        }, 100)
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : '프로젝트 저장에 실패했습니다.'
      alert(`저장 실패: ${message}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handlePrev = () => {
    if (step > 1) {
      setStep(step - 1)
      return
    }

    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative flex max-h-[85vh] w-full max-w-[520px] flex-col overflow-hidden rounded-[32px] bg-white shadow-2xl">
        <div className="flex shrink-0 items-center justify-between border-b border-gray-100 bg-white p-6">
          <h2 className="text-xl font-bold text-zinc-700">프로젝트 목표 설정</h2>
          <div className="text-sm font-medium">
            <span className="text-blue-600">{step}</span>
            <span className="text-zinc-300">/4</span>
          </div>
        </div>

        <div className="scrollbar-hide flex-1 overflow-y-auto px-8 py-10">
          <ProjectForm
            step={step}
            onValidationChange={handleValidationChange}
            onDataChange={handleDataChange}
          />
        </div>

        <div className="shrink-0 border-t border-gray-100 bg-white p-6">
          <div className="flex gap-3">
            <button
              type="button"
              onClick={handlePrev}
              className="flex-1 rounded-full bg-gray-50 py-4 text-sm font-bold text-zinc-400 transition-all hover:bg-gray-100"
            >
              {step === 1 ? '취소' : '이전'}
            </button>

            <button
              type="button"
              onClick={handleNext}
              disabled={!isValid || isSubmitting}
              className={`flex-1 rounded-full py-4 text-sm font-bold transition-all ${
                isValid && !isSubmitting
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-200 hover:bg-blue-700'
                  : 'cursor-not-allowed bg-blue-100 text-white'
              }`}
            >
              {isSubmitting ? '저장 중...' : step === 4 ? '완료' : '다음'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
