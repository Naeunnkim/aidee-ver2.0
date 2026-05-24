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
  demographicInfo?: string[]
  personaStory?: string[]
  problemNeeds?: string[]
  currentBehavior?: string[]
  lifestyleContext?: string[]
  relationshipKeyword?: string[]
  imageUrl?: string
}

type PersonaCardProps = {
  data: PersonaCardData
  onAdjust?: () => void
  onProceed?: () => void
  showActions?: boolean
}

type CardLayoutProps = {
  data: PersonaCardData
  mode: 'screen' | 'export'
}

type CardLayoutWithRefProps = CardLayoutProps & {
  cardRef?: Ref<HTMLDivElement>
}

const PERSONA_CARD_WIDTH = 492
const PERSONA_CARD_HEIGHT = 292
const PERSONA_CARD_DISPLAY_SCALE = 1.35

export default function PersonaCard({
  data,
  onAdjust,
  onProceed,
  showActions = true,
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

        {showActions ? (
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
        ) : null}
      </div>
    </>
  )
}

const CardLayout = ({ data, cardRef }: CardLayoutWithRefProps) => {
  const personaStoryFallback = data.success.map((item) =>
    [item.tag, item.desc].filter(Boolean).join(' - ')
  )
  const sections = [
    {
      title: '(1) Demographic Info',
      items: data.demographicInfo?.length ? data.demographicInfo : data.user,
    },
    {
      title: '(2) Persona Story',
      items: data.personaStory?.length ? data.personaStory : personaStoryFallback,
    },
    {
      title: '(3) Problem & Needs',
      items: data.problemNeeds?.length ? data.problemNeeds : data.problem,
    },
    {
      title: '(4) Current Behavior',
      items: data.currentBehavior?.length
        ? data.currentBehavior
        : data.behaviorMap,
    },
    {
      title: '(5) Lifestyle Context',
      items: data.lifestyleContext?.length
        ? data.lifestyleContext
        : data.decision,
    },
    {
      title: '(6) Relationship Keyword',
      items: data.relationshipKeyword?.length
        ? data.relationshipKeyword
        : data.correlationAnalysis,
    },
  ]

  return (
    <div
      ref={cardRef}
      data-persona-card="true"
      className="relative w-[492px] max-w-full overflow-hidden rounded-xl font-sans shadow-[0px_0px_24px_0px_rgba(0,0,0,0.12)]"
      style={{
        width: `${PERSONA_CARD_WIDTH}px`,
        height: `${PERSONA_CARD_HEIGHT}px`,
        backgroundColor: '#ffffff',
      }}
    >
      <div
        className="absolute left-0 top-0 w-36 rounded-bl-lg rounded-tl-lg"
        style={{
          height: `${PERSONA_CARD_HEIGHT}px`,
          backgroundColor: '#d4d4d8',
        }}
      >
        {data.imageUrl ? (
          <img
            src={data.imageUrl}
            className="absolute inset-0 h-full w-full object-cover"
            alt="Persona"
          />
        ) : null}
      </div>

      <div className="absolute left-[167px] top-[9.5px] flex h-[273px] w-80 flex-col items-start justify-start gap-1 overflow-hidden">
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

        <div className="grid w-[305px] grid-cols-2 gap-x-4 gap-y-1.5">
          {sections.map((section) => (
            <MiniSection
              key={section.title}
              title={section.title}
              items={section.items}
              compact
              maxItems={2}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function MiniSection({
  title,
  items,
  compact = false,
  maxItems = 4,
}: {
  title: string
  items: string[]
  compact?: boolean
  maxItems?: number
}) {
  const visibleItems = items
    .slice(0, maxItems)
    .map((item) => item.replace(/\.{2,}|…/g, '').trim())

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
        {visibleItems.map((item, index) =>
          compact ? (
            <span key={`${title}-${index}-${item}`}>
              • {item}
              {index < visibleItems.length - 1 ? <br /> : null}
            </span>
          ) : (
            <span key={`${title}-${index}-${item}`}>• {item}</span>
          )
        )}
      </div>
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
