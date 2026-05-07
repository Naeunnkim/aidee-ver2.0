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

const PERSONA_CARD_WIDTH = 492
const PERSONA_CARD_HEIGHT = 256
const PERSONA_CARD_DISPLAY_SCALE = 1.35

export default function PersonaCard({
  data,
  onAdjust,
  onProceed,
}: PersonaCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const displayCardRef = useRef<HTMLDivElement>(null)

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
    const target = displayCardRef.current
    if (!target) {
      return
    }

    try {
      if ('fonts' in document) {
        await (document as Document & { fonts: FontFaceSet }).fonts.ready
      }
      await waitForImages(target)

      const exportScale = 3
      const { width, height } = target.getBoundingClientRect()

      const canvas = await html2canvas(target, {
        backgroundColor: '#ffffff',
        useCORS: true,
        scale: exportScale,
        windowWidth: Math.round(width),
        windowHeight: Math.round(height),
        scrollX: 0,
        scrollY: 0,
        imageTimeout: 0,
        logging: false,
        removeContainer: true,
        foreignObjectRendering: false,
      })

      const mimeType = format === 'jpeg' ? 'image/jpeg' : 'image/png'
      const extension = format === 'jpeg' ? 'jpg' : 'png'

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (nextBlob) => {
            if (nextBlob) {
              resolve(nextBlob)
              return
            }

            reject(new Error('Persona card image export failed'))
          },
          mimeType,
          1
        )
      })

      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.download = `persona_card.${extension}`
      link.href = url
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed', error)
    }
  }

  return (
    <>
      <div className="my-4 flex w-full max-w-[664px] flex-col gap-3">
        <div
          ref={displayCardRef}
          className="pointer-events-none origin-top-left scale-[1.35]"
          style={{
            width: `${PERSONA_CARD_WIDTH}px`,
            height: `${PERSONA_CARD_HEIGHT * PERSONA_CARD_DISPLAY_SCALE}px`,
          }}
        >
          <CardLayoutWithRef cardRef={cardRef} data={data} mode="screen" />
        </div>

        <div className="relative z-10 flex gap-2">
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
    </>
  )
}

const CardLayout = ({ data, cardRef }: CardLayoutWithRefProps) => {
  return (
    <div
      ref={cardRef}
      data-persona-card="true"
      className="relative h-64 w-[492px] max-w-full overflow-hidden rounded-xl font-sans shadow-[0px_0px_24px_0px_rgba(0,0,0,0.12)]"
      style={{
        width: `${PERSONA_CARD_WIDTH}px`,
        height: `${PERSONA_CARD_HEIGHT}px`,
        backgroundColor: '#ffffff',
      }}
    >
      <div
        className="absolute left-0 top-0 h-64 w-36 rounded-bl-lg rounded-tl-lg"
        style={{ backgroundColor: '#d4d4d8' }}
      >
        {data.imageUrl ? (
          <img
            src={data.imageUrl}
            className="absolute inset-0 h-full w-full object-cover"
            alt="Persona"
          />
        ) : null}
      </div>

      <div className="absolute left-[167px] top-[9.5px] flex h-[238px] w-80 flex-col items-start justify-start gap-1 overflow-hidden">
        <div className="flex w-72 flex-col items-start justify-start">
          <div className="inline-flex items-end justify-start gap-1">
            <h2
              className="text-center text-lg font-bold"
              style={{ color: '#3f3f46' }}
            >
              Persona Card
            </h2>
          </div>
        </div>

        <div className="inline-flex items-start justify-start gap-5 self-stretch">
          <div className="inline-flex w-32 flex-col items-start justify-start gap-1.5">
            <MiniSection title="User" items={data.user} />
            <MiniSection title="Usage" items={data.behaviorMap} />
            <MiniSection title="Decision" items={data.decision} />
          </div>

          <div className="inline-flex w-36 flex-col items-start justify-start gap-1.5">
            <MiniSection title="Problem" items={data.problem} compact />
            <MiniSection
              title="Current Solution"
              items={data.correlationAnalysis}
              compact
            />
            <SuccessSection items={data.success} />
          </div>
        </div>
      </div>
    </div>
  )
}

function MiniSection({
  title,
  items,
  compact = false,
}: {
  title: string
  items: string[]
  compact?: boolean
}) {
  return (
    <section className="flex w-full flex-col items-start justify-start gap-0.5">
      <div className="flex w-full flex-col items-start justify-start gap-px">
        <h4
          className="text-[8.5px] font-bold leading-3"
          style={{ color: '#2563eb' }}
        >
          {title}
        </h4>
        <div className="h-px w-full" style={{ backgroundColor: '#a1a1aa' }} />
      </div>
      <div
        className={
          compact
            ? 'w-full text-[6.5px] font-semibold leading-[8.5px]'
            : 'flex w-full flex-col items-start justify-start text-[7px] font-semibold leading-[9px]'
        }
        style={{ color: '#3f3f46' }}
      >
        {items.slice(0, 4).map((item, index) =>
          compact ? (
            <span key={`${title}-${index}-${item}`}>
              • {item}
              {index < Math.min(items.length, 4) - 1 ? <br /> : null}
            </span>
          ) : (
            <span key={`${title}-${index}-${item}`}>• {item}</span>
          )
        )}
      </div>
    </section>
  )
}

function SuccessSection({ items }: { items: SuccessItem[] }) {
  return (
    <section className="flex w-full flex-col items-start justify-start gap-0.5">
      <div className="flex w-full flex-col items-start justify-start gap-px">
        <h4
          className="text-center text-[8px] font-bold leading-3"
          style={{ color: '#2563eb' }}
        >
          Success
        </h4>
        <div className="h-px w-full" style={{ backgroundColor: '#a1a1aa' }} />
      </div>

      {items.slice(0, 3).map((item, index) => (
        <div
          key={`${item.tag}-${item.desc}-${index}`}
          className="inline-flex w-full items-center justify-start gap-[3px]"
        >
          <div
            className="flex h-2.5 items-center justify-center rounded-[54.5px] px-1"
            style={{ backgroundColor: '#dbeafe' }}
          >
            <span
              className="max-w-[58px] truncate text-center text-[4.5px] font-medium leading-[6px]"
              style={{ color: '#0ea5e9' }}
            >
              #{item.tag}
            </span>
          </div>
          {item.desc ? (
            <span
              className="flex-1 truncate text-[4.5px] font-medium leading-[6px]"
              style={{ color: '#a1a1aa' }}
            >
              {item.desc}
            </span>
          ) : null}
        </div>
      ))}
    </section>
  )
}

const CardLayoutWithRef = ({
  data,
  mode,
  cardRef,
}: CardLayoutWithRefProps) => {
  return <CardLayout data={data} mode={mode} cardRef={cardRef} />
}
