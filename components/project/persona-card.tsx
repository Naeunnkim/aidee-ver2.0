'use client'

import { useRef, type Ref } from 'react'
import html2canvas from 'html2canvas'

type SuccessItem = {
  tag: string
  desc: string
}

type PersonaCardData = {
  user: string[]
  behaviorMap: string[]
  correlationAnalysis: string[]
  problem: string[]
  decision: string[]
  success: SuccessItem[]
  imageUrl?: string
}

type PersonaCardProps = {
  data: PersonaCardData
  onAdjust?: () => void
  onProceed?: () => void
}

type CardLayoutProps = {
  data: PersonaCardData
  mode: 'screen' | 'export'
}

type CardLayoutWithRefProps = CardLayoutProps & {
  cardRef?: Ref<HTMLDivElement>
}

export default function PersonaCard({
  data,
  onAdjust,
  onProceed,
}: PersonaCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const exportCardRef = useRef<HTMLDivElement>(null)

  const waitForImages = async (root: HTMLElement) => {
    const images = Array.from(root.querySelectorAll('img'))
    await Promise.all(
      images.map((img) => {
        if (img.complete) {
          return Promise.resolve()
        }

        return new Promise<void>((resolve) => {
          img.onload = () => resolve()
          img.onerror = () => resolve()
        })
      })
    )
  }

  const onDownload = async (format: 'png' | 'jpeg' = 'png') => {
    const exportTarget = exportCardRef.current
    if (!exportTarget) {
      return
    }

    try {
      if ('fonts' in document) {
        await (document as Document & { fonts: FontFaceSet }).fonts.ready
      }
      await waitForImages(exportTarget)

      const { width, height } = exportTarget.getBoundingClientRect()
      const exportScale = 3

      const canvas = await html2canvas(exportTarget, {
        backgroundColor: '#ffffff',
        useCORS: true,
        scale: exportScale,
        width: Math.round(width),
        height: Math.round(height),
        windowWidth: Math.round(width),
        windowHeight: Math.round(height),
        imageTimeout: 0,
        logging: false,
        removeContainer: true,
        foreignObjectRendering: false,
      })

      const link = document.createElement('a')
      const mimeType = format === 'jpeg' ? 'image/jpeg' : 'image/png'
      const extension = format === 'jpeg' ? 'jpg' : 'png'
      link.download = `persona_card.${extension}`
      link.href = canvas.toDataURL(mimeType, 1)
      link.click()
    } catch (error) {
      console.error('Download failed', error)
    }
  }

  return (
    <>
      <div className="my-4 flex w-full max-w-[800px] flex-col gap-3">
        <CardLayoutWithRef cardRef={cardRef} data={data} mode="screen" />

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onAdjust?.()}
            className="rounded-full bg-sky-100 px-4 py-1.5 text-sm font-medium text-sky-600 transition-colors hover:bg-sky-200"
          >
            조정하기
          </button>
          <button
            type="button"
            onClick={() => onProceed?.()}
            className="rounded-full bg-blue-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700"
          >
            이대로 진행하기
          </button>
          <button
            type="button"
            onClick={() => onDownload('png')}
            className="ml-auto rounded-full bg-gray-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-black"
          >
            저장하기
          </button>
        </div>
      </div>

      <div
        style={{
          position: 'fixed',
          left: '-10000px',
          top: 0,
          width: '800px',
          zIndex: -1,
          pointerEvents: 'none',
          opacity: 1,
        }}
        aria-hidden="true"
      >
        <CardLayoutWithRef
          cardRef={exportCardRef}
          data={data}
          mode="export"
        />
      </div>
    </>
  )
}

const CardLayout = ({ data, mode, cardRef }: CardLayoutWithRefProps) => {
  const isExport = mode === 'export'

  return (
    <div
      ref={cardRef}
      data-persona-card="true"
        className={`flex w-full rounded-3xl font-sans ${isExport ? '' : 'overflow-hidden'}`}
        style={{
        minHeight: isExport ? 'unset' : '560px',
        height: isExport ? 'auto' : '560px',
        backgroundColor: '#ffffff',
        border: '1px solid #f3f4f6',
        boxShadow: isExport ? 'none' : '0 20px 40px rgba(15, 23, 42, 0.08)',
      }}
    >
      <div
        className="relative w-[30%] shrink-0"
        style={{
          backgroundColor: '#e5e7eb',
          minHeight: isExport ? '640px' : '100%',
        }}
      >
        <img
          src={data.imageUrl || 'https://placehold.co/240x480'}
          className="absolute inset-0 h-full w-full object-cover"
          alt="Persona"
        />
      </div>

      <div
        className={
          isExport
            ? 'flex flex-1 flex-col gap-4 px-8 pt-7 pb-7'
            : 'flex flex-1 flex-col gap-3 overflow-hidden px-7 pt-6 pb-5'
        }
      >
        <div className="flex items-start justify-between">
          <h2
            className={
              isExport
                ? 'text-[30px] leading-tight font-bold tracking-tight'
                : 'pt-0.5 text-xl leading-tight font-bold tracking-tight'
            }
            style={{ color: '#3f3f46' }}
          >
            Persona Card
          </h2>
          <button
            type="button"
            className={
              isExport
                ? 'rounded-full px-4 py-1.5 text-xs font-medium uppercase'
                : '-mt-0.5 rounded-full px-3 py-1 text-[10px] font-medium uppercase'
            }
            style={{ backgroundColor: '#2563eb', color: '#ffffff' }}
          >
            Validated
          </button>
        </div>

        <div
          className={
            isExport
              ? 'space-y-4'
              : 'flex-1 space-y-3 overflow-y-auto pr-1'
          }
        >
          <div
            className={
              isExport
                ? 'grid grid-cols-2 gap-x-8 gap-y-4'
                : 'grid grid-cols-2 gap-x-5 gap-y-3'
            }
          >
            <MiniSection title="User" items={data.user} mode={mode} />
            <MiniSection title="Behavior Map" items={data.behaviorMap} mode={mode} />
            <MiniSection
              title="Correlation Analysis"
              items={data.correlationAnalysis}
              mode={mode}
            />
            <MiniSection title="Problem" items={data.problem} mode={mode} />
            <div className="col-span-2">
              <MiniSection title="Decision" items={data.decision} mode={mode} />
            </div>
          </div>

          <div
            className={isExport ? 'pt-3' : 'pt-2'}
            style={{ borderTop: '1px solid #f3f4f6' }}
          >
            <h4
              className={
                isExport
                  ? 'mb-3 text-xs font-bold uppercase'
                  : 'mb-2 text-[10px] font-bold uppercase'
              }
              style={{ color: '#2563eb' }}
            >
              Success
            </h4>
            <div className={isExport ? 'flex flex-wrap gap-2' : 'flex flex-wrap gap-1.5'}>
              {data.success.map((item, index) => (
                <div
                  key={`${item.tag}-${item.desc}-${index}`}
                  className={
                    isExport
                      ? 'flex items-center gap-2 rounded-full px-3 py-1.5'
                      : 'flex items-center gap-1.5 rounded-full px-2 py-1'
                  }
                  style={{ backgroundColor: '#eff6ff' }}
                >
                  <span
                    className={isExport ? 'text-[11px] font-bold' : 'text-[9px] font-bold'}
                    style={{ color: '#0ea5e9' }}
                  >
                    #{item.tag}
                  </span>
                  {item.desc ? (
                    <span
                      className={
                        isExport
                          ? 'text-[10px] leading-none'
                          : 'text-[8px] leading-none'
                      }
                      style={{ color: '#6b7280' }}
                    >
                      {item.desc}
                    </span>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function MiniSection({
  title,
  items,
  mode,
}: {
  title: string
  items: string[]
  mode: 'screen' | 'export'
}) {
  const isExport = mode === 'export'

  return (
    <div className={isExport ? 'flex flex-col gap-1.5' : 'flex flex-col gap-1'}>
      <div className={isExport ? 'flex flex-col gap-1' : 'flex flex-col gap-0.5'}>
        <h4
          className={
            isExport
              ? 'text-[15px] leading-tight font-bold'
              : 'text-[10px] leading-none font-bold'
          }
          style={{ color: '#2563eb' }}
        >
          {title}
        </h4>
        <div className="w-full" style={{ height: '1px', backgroundColor: '#d4d4d8' }} />
      </div>
      <div className={isExport ? 'flex flex-col gap-0.5' : 'flex flex-col gap-0'}>
        {items.map((item, index) => (
          <p
            key={`${title}-${index}-${item}`}
            className={
              isExport
                ? 'text-[13px] leading-[1.55] font-medium'
                : 'text-[9px] leading-[1.45] font-medium'
            }
            style={{ color: '#3f3f46' }}
          >
            • {item}
          </p>
        ))}
      </div>
    </div>
  )
}

const CardLayoutWithRef = ({
  data,
  mode,
  cardRef,
}: CardLayoutWithRefProps) => {
  return <CardLayout data={data} mode={mode} cardRef={cardRef} />
}
