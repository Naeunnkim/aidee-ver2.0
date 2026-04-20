'use client'

import { useEffect, useRef, useState } from 'react'

interface ProjectFormProps {
  step: number
  onValidationChange: (isValid: boolean) => void
  onDataChange: (data: Record<string, unknown>) => void
}

const TOTAL_MAX = 10000

export default function ProjectForm({
  step,
  onValidationChange,
  onDataChange,
}: ProjectFormProps) {
  const [goal, setGoal] = useState('')
  const [categories, setCategories] = useState<string[]>([])
  const [otherCategory, setOtherCategory] = useState('')
  const [minBudget, setMinBudget] = useState(2000)
  const [maxBudget, setMaxBudget] = useState(7500)

  const [size, setSize] = useState('')
  const [features, setFeatures] = useState<string[]>([])
  const [otherFeature, setOtherFeature] = useState('')
  const [duration, setDuration] = useState('')
  const [usage, setUsage] = useState('')

  const [idea, setIdea] = useState('')

  const [referenceImages, setReferenceImages] = useState<File[]>([])
  const [referencePreviews, setReferencePreviews] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const categoryOptions = [
    '조명',
    '인테리어 소품',
    '가구',
    '패션·악세서리',
    '디지털 기기',
    '기타 (직접 입력)',
  ]

  const featureOptions = [
    '단순 구조물',
    '빛·색 변화',
    '센서 감지',
    '조립·분해 가능',
    'IoT / 스마트 기능',
    '기타 (직접 입력)',
  ]

  const durationOptions = [
    '1주',
    '2주',
    '1개월',
    '3개월',
    '6개월',
    '1년',
    '1년 +',
  ]

  const MAX_REFERENCE_FILES = 4
  const MAX_REFERENCE_FILE_SIZE = 10 * 1024 * 1024
  const ACCEPTED_IMAGE_TYPES = [
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
  ]

  const handleReferenceUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    if (files.length === 0) {
      return
    }

    const validFiles = files.filter(
      (file) =>
        ACCEPTED_IMAGE_TYPES.includes(file.type) &&
        file.size <= MAX_REFERENCE_FILE_SIZE
    )

    const remainingSlots = MAX_REFERENCE_FILES - referenceImages.length
    const filesToAdd = validFiles.slice(0, Math.max(remainingSlots, 0))

    if (filesToAdd.length === 0) {
      e.target.value = ''
      return
    }

    const nextFiles = [...referenceImages, ...filesToAdd]
    const nextPreviews = [
      ...referencePreviews,
      ...filesToAdd.map((file) => URL.createObjectURL(file)),
    ]

    setReferenceImages(nextFiles)
    setReferencePreviews(nextPreviews)
    e.target.value = ''
  }

  const handleReferenceRemove = (index: number) => {
    const previewToRemove = referencePreviews[index]
    if (previewToRemove) {
      URL.revokeObjectURL(previewToRemove)
    }

    setReferenceImages((prev) => prev.filter((_, i) => i !== index))
    setReferencePreviews((prev) => prev.filter((_, i) => i !== index))
  }

  const formatBudget = (value: number) => {
    if (value >= 10000) {
      return '1억 원'
    }

    return `${value.toLocaleString()} 만 원`
  }

  const handleCategoryChange = (cat: string) => {
    setCategories((prev) =>
      prev.includes(cat) ? prev.filter((item) => item !== cat) : [...prev, cat]
    )
  }

  const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Math.min(Number(e.target.value), maxBudget - 500)
    setMinBudget(value)
  }

  const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Math.max(Number(e.target.value), minBudget + 500)
    setMaxBudget(value)
  }

  useEffect(() => {
    let currentValid = false

    if (step === 1) {
      const isOtherValid = categories.includes('기타 (직접 입력)')
        ? otherCategory.trim() !== ''
        : true
      currentValid = goal !== '' && categories.length > 0 && isOtherValid
    } else if (step === 2) {
      const isOtherValid = features.includes('기타 (직접 입력)')
        ? otherFeature.trim() !== ''
        : true
      currentValid =
        size !== '' &&
        features.length > 0 &&
        isOtherValid &&
        duration !== '' &&
        usage !== ''
    } else if (step === 3) {
      currentValid = idea.trim().length > 0
    } else if (step === 4) {
      currentValid = true
    }

    onValidationChange(currentValid)

    onDataChange({
      goal,
      categories,
      otherCategory,
      minBudget,
      maxBudget,
      size,
      features,
      otherFeature,
      duration,
      usage,
      idea,
      referenceImages,
    })
  }, [
    step,
    goal,
    categories,
    otherCategory,
    minBudget,
    maxBudget,
    size,
    features,
    otherFeature,
    duration,
    usage,
    idea,
    referenceImages,
    onValidationChange,
    onDataChange,
  ])

  useEffect(() => {
    return () => {
      referencePreviews.forEach((preview) => URL.revokeObjectURL(preview))
    }
  }, [referencePreviews])

  useEffect(() => {
    const scrollContainer = document.querySelector<HTMLElement>('.overflow-y-auto')
    if (scrollContainer) {
      scrollContainer.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }, [step])

  return (
    <div className="w-full">
      {step === 1 && (
        <div className="flex flex-col gap-12 pb-10">
          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                1. 제품 개발의 어느 단계까지 목표로 하고 계신가요?
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                선택한 단계에 따라 추천 프로세스를 맞춤 제공해드려요.
              </p>
            </div>
            <div className="flex flex-col gap-1 pl-2">
              {['아이디어 구체화', '2D·3D 시각화', '시제품 제작 및 사업화'].map(
                (option) => (
                  <label
                    key={option}
                    className="group flex h-10 cursor-pointer items-center gap-2.5 rounded-lg px-2 transition-colors hover:bg-slate-50"
                  >
                    <div className="relative flex h-5 w-5 items-center justify-center">
                      <input
                        type="radio"
                        name="goal"
                        value={option}
                        checked={goal === option}
                        onChange={(e) => setGoal(e.target.value)}
                        className="peer hidden"
                      />
                      <div className="h-4 w-4 rounded-full border-2 border-gray-200 transition-all peer-checked:border-blue-600" />
                      <div className="absolute h-1.5 w-1.5 rounded-full bg-blue-600 opacity-0 transition-all peer-checked:opacity-100" />
                    </div>
                    <span
                      className={`text-sm font-medium transition-colors ${
                        goal === option ? 'text-blue-600' : 'text-slate-500'
                      }`}
                    >
                      {option}
                    </span>
                  </label>
                )
              )}
            </div>
          </section>

          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                2. 디자인·개발하고자 하는 제품 카테고리는 무엇인가요?
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                전체 제품 흐름과 참조 데이터를 설계하는 데 사용돼요.
              </p>
            </div>
            <div className="flex flex-col gap-1 pl-2">
              {categoryOptions.map((cat) => (
                <div key={cat} className="flex flex-col gap-2">
                  <label className="flex h-9 cursor-pointer items-center gap-2.5 rounded-lg px-2 transition-colors hover:bg-slate-50">
                    <div className="relative flex h-5 w-5 items-center justify-center">
                      <input
                        type="checkbox"
                        checked={categories.includes(cat)}
                        onChange={() => handleCategoryChange(cat)}
                        className="peer hidden"
                      />
                      <div className="h-4 w-4 rounded-[4px] border-2 border-gray-200 transition-all peer-checked:border-blue-600 peer-checked:bg-blue-600" />
                      <div className="absolute text-[10px] text-white opacity-0 peer-checked:opacity-100">
                        ✓
                      </div>
                    </div>
                    <span
                      className={`text-sm font-medium transition-colors ${
                        categories.includes(cat)
                          ? 'text-blue-600'
                          : 'text-slate-500'
                      }`}
                    >
                      {cat}
                    </span>
                  </label>
                  {cat === '기타 (직접 입력)' && categories.includes(cat) && (
                    <div className="animate-in fade-in zoom-in-95 px-4 pb-2 pl-10">
                      <input
                        type="text"
                        placeholder="카테고리를 직접 입력해주세요"
                        value={otherCategory}
                        onChange={(e) => setOtherCategory(e.target.value)}
                        className="h-10 w-full rounded-xl border border-slate-100 bg-slate-50 px-4 text-sm outline-none transition-all focus:border-blue-300"
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="flex flex-col gap-8">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                3. 예상 예산 범위를 알려주세요
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                예산에 맞춰 실현 가능한 설계 전략을 제안해드려요.
              </p>
            </div>
            <div className="flex flex-col gap-10 px-2">
              <div className="relative h-1.5 w-full rounded-full bg-gray-200">
                <div
                  className="absolute h-full rounded-full bg-blue-600"
                  style={{
                    left: `${(minBudget / TOTAL_MAX) * 100}%`,
                    right: `${100 - (maxBudget / TOTAL_MAX) * 100}%`,
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max={TOTAL_MAX}
                  step="500"
                  value={minBudget}
                  onChange={handleMinChange}
                  className="custom-slider-handle pointer-events-none absolute z-20 h-1.5 w-full appearance-none bg-transparent accent-blue-600"
                />
                <input
                  type="range"
                  min="0"
                  max={TOTAL_MAX}
                  step="500"
                  value={maxBudget}
                  onChange={handleMaxChange}
                  className="custom-slider-handle pointer-events-none absolute z-20 h-1.5 w-full appearance-none bg-transparent accent-blue-600"
                />
              </div>
              <div className="relative flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] leading-4 font-medium text-zinc-400">
                    최소
                  </span>
                  <div className="rounded-lg border border-gray-100 bg-white px-3 py-1.5 shadow-sm">
                    <span className="text-sm font-medium text-neutral-900">
                      {formatBudget(minBudget)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] leading-4 font-medium text-zinc-400">
                    최대
                  </span>
                  <div className="rounded-lg border border-gray-100 bg-white px-3 py-1.5 shadow-sm">
                    <span className="text-sm font-medium text-neutral-900">
                      {formatBudget(maxBudget)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <style jsx>{`
            .custom-slider-handle::-webkit-slider-thumb {
              pointer-events: auto;
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #2563eb;
              border: 4px solid white;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
              cursor: pointer;
              appearance: none;
            }
            .custom-slider-handle::-moz-range-thumb {
              pointer-events: auto;
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #2563eb;
              border: 4px solid white;
              cursor: pointer;
            }
          `}</style>
        </div>
      )}

      {step === 2 && (
        <div className="animate-in fade-in slide-in-from-right-4 flex flex-col gap-12 pb-10 duration-500">
          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                4. 제품의 예상 크기를 알려주세요
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                크기에 따라 재료/공정/예산을 달리 제안해요.
              </p>
            </div>
            <div className="flex flex-col gap-1 pl-2">
              {[
                '손바닥 크기 (10cm 이내)',
                '소형 (10~50cm)',
                '중형 (50~100cm)',
                '대형 (100cm 이상)',
                '아직 못 정했어요',
              ].map((opt) => (
                <label
                  key={opt}
                  className="group flex h-10 cursor-pointer items-center gap-2.5 rounded-lg px-2 transition-colors hover:bg-slate-50"
                >
                  <div className="relative flex h-5 w-5 items-center justify-center">
                    <input
                      type="radio"
                      name="size"
                      value={opt}
                      checked={size === opt}
                      onChange={(e) => setSize(e.target.value)}
                      className="peer hidden"
                    />
                    <div className="h-4 w-4 rounded-full border-2 border-gray-200 transition-all peer-checked:border-blue-600" />
                    <div className="absolute h-1.5 w-1.5 rounded-full bg-blue-600 opacity-0 transition-all peer-checked:opacity-100" />
                  </div>
                  <span
                    className={`text-sm font-medium transition-colors ${
                      size === opt ? 'text-blue-600' : 'text-slate-500'
                    }`}
                  >
                    {opt}
                  </span>
                </label>
              ))}
            </div>
          </section>

          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                5. 어떤 기능을 포함하고 싶으신가요?
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                제품 기능의 방향성을 반영해 아이디어 확장을 도와드려요.
              </p>
            </div>
            <div className="flex flex-col gap-1 pl-2">
              {featureOptions.map((feat) => (
                <div key={feat} className="flex flex-col gap-2">
                  <label className="flex h-9 cursor-pointer items-center gap-2.5 rounded-lg px-2 transition-colors hover:bg-slate-50">
                    <div className="relative flex h-5 w-5 items-center justify-center">
                      <input
                        type="checkbox"
                        checked={features.includes(feat)}
                        onChange={() =>
                          setFeatures((prev) =>
                            prev.includes(feat)
                              ? prev.filter((f) => f !== feat)
                              : [...prev, feat]
                          )
                        }
                        className="peer hidden"
                      />
                      <div className="h-4 w-4 rounded-[4px] border-2 border-gray-200 transition-all peer-checked:border-blue-600 peer-checked:bg-blue-600" />
                      <div className="absolute text-[10px] text-white opacity-0 peer-checked:opacity-100">
                        ✓
                      </div>
                    </div>
                    <span
                      className={`text-sm font-medium transition-colors ${
                        features.includes(feat)
                          ? 'text-blue-600'
                          : 'text-slate-500'
                      }`}
                    >
                      {feat}
                    </span>
                  </label>

                  {feat === '기타 (직접 입력)' && features.includes(feat) && (
                    <div className="animate-in fade-in zoom-in-95 px-4 pb-2 pl-10">
                      <input
                        type="text"
                        placeholder="기능을 직접 입력해주세요"
                        value={otherFeature}
                        onChange={(e) => setOtherFeature(e.target.value)}
                        className="h-10 w-full rounded-xl border border-slate-100 bg-slate-50 px-4 text-sm outline-none transition-all focus:border-blue-300"
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                6. 제품을 언제까지 완성하고 싶으신가요?
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                해당 기간 내 목표 달성을 위한 프로세스를 자동 제안해 드려요.
              </p>
            </div>
            <div className="flex flex-wrap gap-1.5 pl-2">
              {durationOptions.map((dur) => (
                <button
                  key={dur}
                  type="button"
                  onClick={() => setDuration(dur)}
                  className={`rounded-full px-3.5 py-1.5 text-xs font-semibold transition-all ${
                    duration === dur
                      ? 'bg-blue-100 text-blue-600 outline outline-1 outline-blue-600'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                  }`}
                >
                  {dur}
                </button>
              ))}
            </div>
          </section>

          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                7. 사용 용도가 어떻게 되시나요?
              </h3>
              <p className="pl-6 text-xs font-medium text-gray-400">
                목적에 맞춰 최적화된 프로세스로 제안해드려요.
              </p>
            </div>
            <div className="flex flex-col gap-1 pl-2">
              {[
                '개인 소장 및 전시용',
                '대량 판매',
                '크라우드 펀딩',
                '브랜드 런칭',
              ].map((u) => (
                <label
                  key={u}
                  className="group flex h-10 cursor-pointer items-center gap-2.5 rounded-lg px-2 transition-colors hover:bg-slate-50"
                >
                  <div className="relative flex h-5 w-5 items-center justify-center">
                    <input
                      type="radio"
                      name="usage"
                      value={u}
                      checked={usage === u}
                      onChange={(e) => setUsage(e.target.value)}
                      className="peer hidden"
                    />
                    <div className="h-4 w-4 rounded-full border-2 border-gray-200 transition-all peer-checked:border-blue-600" />
                    <div className="absolute h-1.5 w-1.5 rounded-full bg-blue-600 opacity-0 transition-all peer-checked:opacity-100" />
                  </div>
                  <span
                    className={`text-sm font-medium transition-colors ${
                      usage === u ? 'text-blue-600' : 'text-slate-500'
                    }`}
                  >
                    {u}
                  </span>
                </label>
              ))}
            </div>
          </section>
        </div>
      )}

      {step === 3 && (
        <div className="animate-in fade-in slide-in-from-right-4 flex flex-col gap-5 duration-500">
          <div className="flex flex-col gap-0.5">
            <div className="inline-flex items-center gap-2">
              <h3 className="text-lg leading-7 font-semibold text-slate-600">
                Q. 지금 떠오르는 제품 아이디어가 있다면 자유롭게 적어주세요
              </h3>
            </div>
            <div className="pl-5">
              <p className="text-xs leading-5 font-medium text-gray-400">
                어떤 느낌인지, 누구를 위한 제품인지, 어떤 목적으로 만들고 싶은지
                등을 적어주세요.
              </p>
            </div>
          </div>

          <div className="mx-auto h-[392px] w-full max-w-[480px]">
            <textarea
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="감성적인 무드등을 만들고 싶어요. 20대들의 홈 인테리어용으로 개발해서 판매까지 하는 것을 생각 중인데..."
              className="h-full w-full resize-none rounded-2xl border border-gray-200 p-5 text-sm leading-5 font-medium text-neutral-900 outline-none transition-all placeholder:text-gray-300 focus:border-blue-500"
            />
          </div>
        </div>
      )}

      {step === 4 && (
        <div className="animate-in fade-in slide-in-from-right-4 flex flex-col gap-6 pb-6 duration-500">
          <section className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
              <div className="flex flex-col gap-1">
                <h3 className="text-lg leading-7 font-semibold text-slate-600">
                  9. 참고 이미지가 있다면 업로드해주세요.
                </h3>
                <p className="pl-1 text-xs font-medium text-gray-400">
                  최대 {MAX_REFERENCE_FILES}개 · 파일당 최대 용량 10MB · 지원 형식:
                  PNG, JPG, JPEG, WEBP
                </p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              multiple
              onChange={handleReferenceUpload}
              className="hidden"
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex min-h-[180px] w-full flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-6 py-8 text-center transition-colors hover:bg-slate-100"
            >
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-200 bg-white text-2xl shadow-sm">
                🖼️
              </div>
              <div className="flex flex-col gap-1">
                <p className="text-sm font-semibold text-slate-500">
                  이미지 파일을 드래그하거나 클릭해서 추가해주세요.
                </p>
                <p className="text-xs font-medium text-slate-400">
                  무드보드, 레퍼런스 제품, 컬러/재질 참고 이미지 모두 가능해요.
                </p>
              </div>
            </button>
          </section>

          {referencePreviews.length > 0 && (
            <section className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-slate-600">
                  업로드된 참고 이미지
                </h4>
                <span className="text-xs font-medium text-slate-400">
                  {referencePreviews.length} / {MAX_REFERENCE_FILES}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {referencePreviews.map((preview, index) => (
                  <div
                    key={`${preview}-${index}`}
                    className="group relative aspect-square overflow-hidden rounded-2xl border border-slate-200 bg-white"
                  >
                    <img
                      src={preview}
                      alt={`reference-${index + 1}`}
                      className="h-full w-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => handleReferenceRemove(index)}
                      className="absolute top-2 right-2 h-8 w-8 rounded-full bg-black/60 text-sm text-white opacity-0 transition-opacity group-hover:opacity-100"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  )
}
