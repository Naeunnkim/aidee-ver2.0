'use client'

import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'

import {
  extractGeneratedImagesBlock,
  type GeneratedImageBlock,
} from '@/lib/image-generation'
import {
  EXPERT_DEFINITIONS,
  getExpertDefinition,
  isExpertKey,
  type ExpertKey,
} from '@/lib/experts'
import PersonaCard from '@/components/project/persona-card'
import { type RfpDocument, extractRfpJsonBlock } from '@/lib/rfp'
import { saveGeneratedProjectThumbnail } from '@/lib/project-thumbnail'
import { createClient } from '@/lib/supabase/client'
import {
  SIDEBAR_STEPS,
  STAGE_DEFINITIONS,
  getNextStageKey,
  getSidebarStepIndex,
  isKnownStageKey,
  type StageKey,
} from '@/lib/study'

type ChatMessage = {
  id: string
  role: string
  content: string
  seq_order?: number
  active_agent?: string | null
  created_at?: string | null
  stage_key?: StageKey
  generatedImages?: string[]
  generatedImagePrompt?: string | null
}

type ChatApiMessage = {
  role: string
  content: string
}

type StageTimelineItem = {
  stage_key: StageKey
  entered_at: string
  exited_at: string | null
  stage_order: number
}

type CompanyRecommendation = {
  name: string
  summary: string
  website: string
  highlight: string
}

function normalizeAssistantMessage(message: ChatMessage) {
  const { cleanedText: withoutImages, imageBlock } = extractGeneratedImagesBlock(
    message.content
  )
  const { cleanedText, rfpJson } = extractRfpJsonBlock(withoutImages)

  return {
    normalizedMessage: {
      ...message,
      content: cleanedText,
      generatedImages: imageBlock?.images ?? [],
      generatedImagePrompt: imageBlock?.prompt ?? null,
    } satisfies ChatMessage,
    imageBlock,
    rfpJson,
  }
}

function buildChatApiMessages(messages: ChatMessage[]): ChatApiMessage[] {
  return messages.map((message) => {
    const { cleanedText } = extractGeneratedImagesBlock(message.content)

    return {
      role: message.role,
      content: cleanedText,
    }
  })
}

function parsePersonaData(content: string) {
  const normalizedContent = content.replace(/\r\n/g, '\n')

  const imageUrlMatch = normalizedContent.match(
    /https?:\/\/[^\s]+(?:\.jpg|\.jpeg|\.png|\.webp|\/seed\/[^\s]+)/i
  )
  const imageUrl = imageUrlMatch ? imageUrlMatch[0] : ''

  const sectionPatterns = [
    '1\\.\\s*User',
    'User',
    '2\\.\\s*Behavior\\s*Map',
    'Behavior\\s*Map',
    '3\\.\\s*Correlation\\s*Analysis',
    'Correlation\\s*Analysis',
    '4\\.\\s*Problem',
    'Problem',
    'Success',
    '5\\.\\s*Decision',
    'Decision',
  ]

  const cleanLine = (line: string) =>
    line
      .replace(/^#{1,6}\s*/, '')
      .replace(/^\*\*\s*/, '')
      .replace(/\s*\*\*$/, '')
      .replace(/^[-•]\s*/, '')
      .trim()

  const isPersonaNoiseLine = (line: string) =>
    /^\[?(리서치 진행|페르소나 수정|이대로 진행|조정하기|저장하기)\]?$/.test(
      line
    ) ||
    /이미지를 생성했습니다|아래 시안|수정 방향|페르소나의 프로필|카드 블록 종료/.test(
      line
    ) ||
    /이 페르소나로.*리서치|리서치를 진행할까요|아니면 페르소나|페르소나를 수정할까요/.test(
      line
    )

  const trimCardValue = (line: string) =>
    line
      .replace(/^\[(리서치 진행|페르소나 수정)\]$/, '')
      .replace(/^핵심 문제\s*\(.+\)$/i, '')
      .replace(/^핵심 문제\s*3개만 작성,?\s*/i, '')
      .replace(/^핵심 문제\s*:?/i, '')
      .replace(/^[-•]\s*/, '')
      .replace(/\s+/g, ' ')
      .trim()

  const isTemplateGuideValue = (line: string) =>
    /^\(.+\)$/.test(line) ||
    /\d+자 이내|서술형 금지|요약\s*\/|실제 성공 키워드|실제 기대 변화|실제 재사용 이유/.test(
      line
    )

  const normalizeBehaviorMap = (lines: string[]) => {
    const normalized: string[] = []

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]
      const stepMatch = line.match(/^Step\s*(\d+)\s*:\s*(.*)$/i)

      if (!stepMatch) {
        normalized.push(line)
        continue
      }

      const stepNumber = stepMatch[1]
      const rawValue = stepMatch[2].trim()
      const nextLine = lines[index + 1]?.trim()
      const useNextLine =
        (!rawValue || isTemplateGuideValue(rawValue)) &&
        nextLine &&
        !/^Step\s*\d+\s*:/i.test(nextLine) &&
        !isTemplateGuideValue(nextLine)

      if (useNextLine) {
        normalized.push(`Step ${stepNumber}: ${nextLine}`)
        index += 1
        continue
      }

      if (rawValue && !isTemplateGuideValue(rawValue)) {
        normalized.push(`Step ${stepNumber}: ${rawValue}`)
      }
    }

    return normalized
  }

  const filterTemplateGuideLines = (lines: string[]) =>
    lines.filter((line) => !isTemplateGuideValue(line))

  const normalizeLines = (raw: string) => {
    const lines = raw
      .split('\n')
      .map(cleanLine)
      .map(trimCardValue)
      .filter(Boolean)
      .filter((line) => !isPersonaNoiseLine(line))
      .filter((line) => !/^```/.test(line))
      .filter((line) => !/^[-–—]+$/.test(line))

    const merged: string[] = []

    for (const line of lines) {
      if (line.startsWith(':')) {
        const value = line.replace(/^:\s*/, '').trim()
        if (merged.length > 0 && value) {
          merged[merged.length - 1] = `${merged[merged.length - 1]}: ${value}`
        }
        continue
      }

      if (/^\(.+\)$/.test(line) && merged.length > 0) {
        merged[merged.length - 1] = `${merged[merged.length - 1]} ${line}`
        continue
      }

      merged.push(line)
    }

    return merged
  }

  const extractSection = (titlePattern: string) => {
    const nextTitles = sectionPatterns
      .filter((item) => item !== titlePattern)
      .join('|')
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:##+\\s*)?(?:\\*\\*\\s*)?${titlePattern}(?:\\s*\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:##+\\s*)?(?:\\*\\*\\s*)?(?:${nextTitles})(?:\\s*\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalizedContent.match(regex)
    if (!match) {
      return []
    }

    return normalizeLines(match[1])
  }

  const successLines = extractSection('Success')
  const genericSuccessLabels = new Set(['핵심가치', '기대효과', '기대효과2'])
  const successData = successLines
    .map((line, index, lines) => {
      if (line.startsWith('#')) {
        const tag = line.replace(/^#/, '').trim()
        const desc = lines[index + 1]?.startsWith('#')
          ? ''
          : (lines[index + 1] ?? '')

        if (genericSuccessLabels.has(tag)) {
          return desc ? { tag: desc, desc: '' } : null
        }

        return {
          tag,
          desc,
        }
      }

      if (index > 0 && lines[index - 1]?.startsWith('#')) {
        return null
      }

      const [tag, ...descParts] = line.split('→').map((s) => s.trim())
      const normalizedTag = tag.replace(/^#/, '').trim()
      const desc = descParts.join(' ')

      if (genericSuccessLabels.has(normalizedTag)) {
        return desc ? { tag: desc, desc: '' } : null
      }

      return {
        tag: normalizedTag,
        desc,
      }
    })
    .filter(
      (item): item is { tag: string; desc: string } =>
        Boolean(item?.tag || item?.desc)
    )

  const userLines = extractSection('1\\.\\s*User').length
    ? extractSection('1\\.\\s*User')
    : extractSection('User')
  const behaviorMapLines = extractSection('2\\.\\s*Behavior\\s*Map').length
    ? extractSection('2\\.\\s*Behavior\\s*Map')
    : extractSection('Behavior\\s*Map')
  const correlationAnalysisLines = extractSection(
    '3\\.\\s*Correlation\\s*Analysis'
  ).length
    ? extractSection('3\\.\\s*Correlation\\s*Analysis')
    : extractSection('Correlation\\s*Analysis')
  const problemLines = extractSection('4\\.\\s*Problem').length
    ? extractSection('4\\.\\s*Problem')
    : extractSection('Problem')
  const decisionLines = extractSection('5\\.\\s*Decision').length
    ? extractSection('5\\.\\s*Decision')
    : extractSection('Decision')

  const parsed = {
    user: filterTemplateGuideLines(userLines),
    behaviorMap: normalizeBehaviorMap(behaviorMapLines),
    correlationAnalysis: filterTemplateGuideLines(correlationAnalysisLines),
    problem: filterTemplateGuideLines(problemLines),
    success: successData,
    decision: filterTemplateGuideLines(decisionLines),
    imageUrl,
  }

  const hasMinimumData =
    parsed.user.length > 0 ||
    parsed.behaviorMap.length > 0 ||
    parsed.correlationAnalysis.length > 0 ||
    parsed.problem.length > 0 ||
    parsed.decision.length > 0 ||
    parsed.success.length > 0

  return hasMinimumData ? parsed : null
}

function ExpertAvatar({
  expertKey,
  selected = false,
  className = '',
}: {
  expertKey: ExpertKey
  selected?: boolean
  className?: string
}) {
  const mutedClass = selected ? '' : 'opacity-45 grayscale'

  if (expertKey === 'planner') {
    return (
      <span
        className={`relative block h-6 w-6 overflow-hidden rounded-full bg-indigo-50 ${mutedClass} ${className}`}
      >
        <span className="absolute left-[7px] top-[4px] h-3 w-2.5 rounded-t-full bg-indigo-300" />
        <span className="absolute left-[9px] top-[15px] h-1 w-1.5 rounded-full bg-blue-600" />
      </span>
    )
  }

  if (expertKey === 'style_designer') {
    return (
      <span
        className={`relative block h-6 w-6 overflow-hidden rounded-full bg-fuchsia-50 ${mutedClass} ${className}`}
      >
        <span className="absolute left-[6px] top-[5px] h-2.5 w-2.5 rounded-full bg-[radial-gradient(ellipse_at_top_left,_#4E46E2_20%,_#625DF6_55%,_#E37DFF_95%)]" />
        <span className="absolute left-[4px] top-[9px] h-2.5 w-1 rounded-full bg-gradient-to-b from-sky-500 to-green-400" />
        <span className="absolute left-[11px] top-[10px] h-1.5 w-1.5 rounded-full bg-gradient-to-l from-violet-700 to-fuchsia-400" />
      </span>
    )
  }

  if (expertKey === 'engineer') {
    return (
      <span
        className={`relative block h-6 w-6 overflow-hidden rounded-full bg-emerald-100 ${mutedClass} ${className}`}
      >
        <span className="absolute left-[4px] top-[4px] h-4 w-4 rotate-45 rounded-sm bg-green-400" />
        <span className="absolute left-[7px] top-[6px] h-2.5 w-2.5 rounded-sm bg-green-500" />
        <span className="absolute left-[10px] top-[10px] h-1 w-1 rounded-full bg-green-200" />
      </span>
    )
  }

  if (expertKey === 'marketer') {
    return (
      <span
        className={`relative block h-6 w-6 overflow-hidden rounded-full bg-sky-50 ${mutedClass} ${className}`}
      >
        <span className="absolute left-[5px] top-[8px] h-2.5 w-3.5 rounded-sm bg-gradient-to-b from-sky-200 to-fuchsia-300" />
        <span className="absolute left-[7px] top-[13px] h-1.5 w-0.5 rounded-full bg-sky-500" />
        <span className="absolute left-[11px] top-[10px] h-2.5 w-0.5 rounded-full bg-sky-600" />
        <span className="absolute left-[15px] top-[12px] h-2 w-0.5 rounded-full bg-sky-400" />
      </span>
    )
  }

  return (
    <span
      className={`flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-[10px] font-semibold text-blue-700 ${className}`}
    >
      A
    </span>
  )
}

function getStageExperts(stageKey: StageKey): ExpertKey[] {
  switch (stageKey) {
    case 'step_1_idea':
      return ['planner']
    case 'step_2_persona':
    case 'step_2_research':
      return ['planner', 'marketer']
    case 'step_3_direction':
      return ['planner', 'style_designer', 'engineer', 'marketer']
    case 'step_4_style':
    case 'step_4_definition':
      return ['style_designer']
    case 'step_5_design':
      return ['style_designer', 'engineer']
    case 'step_6_rfp':
    case 'step_6_company':
    case 'step_5_rfp':
      return ['planner', 'engineer', 'marketer', 'style_designer']
    default:
      return ['planner']
  }
}

function getStageLabel(stageKey: StageKey) {
  return (
    STAGE_DEFINITIONS.find((stage) => stage.key === stageKey)?.sidebarLabel ??
    '제품 아이디어&개발 조건 정리'
  )
}

function getStageSignature(stageKey: StageKey) {
  return `${stageKey}:${getStageExperts(stageKey).join(',')}`
}

function StageDivider({ stageKey }: { stageKey: StageKey }) {
  const experts = getStageExperts(stageKey)
  const expertLabels = experts
    .map((expert) => getExpertDefinition(expert).label)
    .join(', ')

  return (
    <div className="flex items-center gap-2 py-1">
      <div className="h-px flex-1 bg-gray-100" />
      <div className="inline-flex max-w-[78%] items-center gap-2 rounded-full bg-white px-3 py-1.5 shadow-sm outline outline-1 outline-gray-100">
        <div className="flex -space-x-1">
          {experts.map((expert) => (
            <ExpertAvatar
              key={expert}
              expertKey={expert}
              selected
              className="h-5 w-5 ring-2 ring-white"
            />
          ))}
        </div>
        <span className="truncate text-xs font-medium text-slate-500">
          {getStageLabel(stageKey)} · 참여 전문가: {expertLabels}
        </span>
      </div>
      <div className="h-px flex-1 bg-gray-100" />
    </div>
  )
}

function buildCompanyRecommendations(
  rfpJson: RfpDocument | null,
  rfpContent: string | null
): CompanyRecommendation[] {
  const styleHint =
    rfpJson?.styleKeywords.slice(0, 2).join(' · ') ||
    (rfpContent ? 'RFP 기준' : '프로젝트 기준')
  const targetHint =
    rfpJson?.mainTarget.slice(0, 20) ||
    rfpJson?.projectGoal.slice(0, 20) ||
    '프로젝트 목적'

  return [
    {
      name: 'Intenxiv',
      summary: `${styleHint} 기반 디자인 고도화`,
      website: 'example.com',
      highlight: `핵심: ${targetHint}`,
    },
    {
      name: '상상제작소',
      summary: `${styleHint} 시제품 제작 / 목업 대응`,
      website: 'example.com',
      highlight: '핵심: 기구·회로·제작',
    },
    {
      name: '(주) 한국기술',
      summary: `${styleHint} 양산성 검토 / 제작 파트너`,
      website: 'example.com',
      highlight: '핵심: 제작 안정성',
    },
  ]
}

function CompanyRecommendationsPanel({
  rfpJson,
  rfpContent,
}: {
  rfpJson: RfpDocument | null
  rfpContent: string | null
}) {
  const companies = buildCompanyRecommendations(rfpJson, rfpContent)

  return (
    <div className="flex flex-col gap-4">
      <div className="max-w-[602px] rounded-[24px] rounded-tl-none bg-gray-200 p-5 text-base leading-relaxed font-medium text-neutral-900">
        마지막은 시제품 제작에 필요한 협력 업체를 추천해드리는 시간입니다.
        앞서 정리한 요구사항과 디자인 방향을 기준으로 3곳을 제안합니다.
      </div>

      <div className="flex flex-wrap gap-5">
        {companies.map((company) => (
          <div
            key={company.name}
            className="w-[240px] overflow-hidden rounded-2xl border border-gray-300 bg-white shadow-sm"
          >
            <div className="flex h-36 items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
              <div className="flex h-20 w-44 items-center justify-center rounded-2xl bg-white/80 text-sm font-semibold text-slate-300">
                thumbnail
              </div>
            </div>
            <div className="bg-gray-200 p-3">
              <div className="flex flex-col gap-1.5">
                <div className="text-base font-semibold leading-6 text-neutral-900">
                  {company.name}
                </div>
                <div className="line-clamp-1 text-xs font-medium leading-4 text-slate-500">
                  {company.summary}
                </div>
                <div className="line-clamp-1 text-xs font-medium leading-4 text-slate-400">
                  {company.highlight}
                </div>
                <div className="text-xs font-medium leading-4 text-blue-500 underline">
                  {company.website}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <button
        type="button"
        className="inline-flex w-fit items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-200"
      >
        업체 더 보기
        <span className="text-blue-500">›</span>
      </button>
    </div>
  )
}

function StageAdvanceButton({
  label,
  onClick,
  disabled,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-blue-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {label}
    </button>
  )
}

function RfpActionPanel({
  isDownloadingRfp,
  isLoading,
  onDownload,
  onCompanyConnect,
}: {
  isDownloadingRfp: boolean
  isLoading: boolean
  onDownload: () => void
  onCompanyConnect: () => void
}) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onDownload}
        disabled={isDownloadingRfp || isLoading}
        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isDownloadingRfp ? 'RFP 생성 중...' : 'RFP 다운로드'}
      </button>
      <button
        type="button"
        onClick={onCompanyConnect}
        disabled={isLoading}
        className="rounded-full bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-60"
      >
        협력업체 연결
      </button>
    </div>
  )
}

export default function ChatPage({
  projectId,
  projectTitle,
  userName,
}: {
  projectId: string
  projectTitle: string
  userName: string
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isInitialLoading, setIsInitialLoading] = useState(true)
  const [isDownloadingRfp, setIsDownloadingRfp] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [currentStageKey, setCurrentStageKey] = useState<StageKey>('step_1_idea')
  const [activeExpert, setActiveExpert] = useState<ExpertKey>('aidee')
  const [latestRfpJson, setLatestRfpJson] = useState<RfpDocument | null>(null)
  const [latestRfpContent, setLatestRfpContent] = useState<string | null>(null)
  const [latestGeneratedImageBlock, setLatestGeneratedImageBlock] =
    useState<GeneratedImageBlock | null>(null)
  const [stageTimeline, setStageTimeline] = useState<StageTimelineItem[]>([])
  const [selectedGeneratedImages, setSelectedGeneratedImages] = useState<
    Record<string, number>
  >({})
  const [isExpertPickerOpen, setIsExpertPickerOpen] = useState(false)

  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const searchParams = useSearchParams()
  const activeExpertDefinition = getExpertDefinition(activeExpert)
  const selectableExperts = EXPERT_DEFINITIONS.filter(
    (expert) => expert.key !== 'aidee'
  )
  const currentStageExperts = getStageExperts(currentStageKey)
  const currentStageExpertLabels = currentStageExperts
    .map((expert) => getExpertDefinition(expert).label)
    .join(', ')
  const expertSuggestionCandidates = selectableExperts.filter(
    (expert) => !currentStageExperts.includes(expert.key)
  )
  const expertRepliesByAideeId = new Map<string, ChatMessage[]>()
  const nestedExpertMessageIds = new Set<string>()
  let latestRenderableAideeId: string | null = null

  for (const message of messages) {
    if (message.role === 'user') {
      latestRenderableAideeId = null
      continue
    }

    if (message.role !== 'assistant') {
      continue
    }

    if (!message.active_agent || message.active_agent === 'aidee') {
      latestRenderableAideeId = message.id
      continue
    }

    if (isExpertKey(message.active_agent) && latestRenderableAideeId) {
      const currentReplies =
        expertRepliesByAideeId.get(latestRenderableAideeId) ?? []
      currentReplies.push(message)
      expertRepliesByAideeId.set(latestRenderableAideeId, currentReplies)
      nestedExpertMessageIds.add(message.id)
    }
  }

  const latestAideeAssistantId = [...messages]
    .reverse()
    .find(
      (message) =>
        message.role === 'assistant' &&
        (!message.active_agent || message.active_agent === 'aidee')
    )?.id

  const getMessageStageKey = (message: ChatMessage): StageKey => {
    if (message.stage_key) {
      return message.stage_key
    }

    if (!message.created_at || stageTimeline.length === 0) {
      return currentStageKey
    }

    const messageTime = new Date(message.created_at).getTime()
    const exactStage = stageTimeline.find((stage) => {
      const enteredAt = new Date(stage.entered_at).getTime()
      const exitedAt = stage.exited_at
        ? new Date(stage.exited_at).getTime()
        : Number.POSITIVE_INFINITY

      return messageTime >= enteredAt && messageTime <= exitedAt
    })

    if (exactStage) {
      return exactStage.stage_key
    }

    const nearestPreviousStage = [...stageTimeline]
      .reverse()
      .find((stage) => new Date(stage.entered_at).getTime() <= messageTime)

    return nearestPreviousStage?.stage_key ?? currentStageKey
  }

  const getResponseStageKey = (
    response: Response,
    fallbackStageKey: StageKey
  ) => {
    const currentStageHeader = response.headers.get('x-aidee-current-stage')
    const nextStageHeader = response.headers.get('x-aidee-next-stage')

    if (currentStageHeader && isKnownStageKey(currentStageHeader)) {
      return currentStageHeader
    }

    if (nextStageHeader && isKnownStageKey(nextStageHeader)) {
      return nextStageHeader
    }

    return fallbackStageKey
  }

  const latestRenderableMessage = [...messages]
    .reverse()
    .find((message) => !nestedExpertMessageIds.has(message.id))
  const shouldShowCurrentStageDivider =
    Boolean(latestRenderableMessage) &&
    latestRenderableMessage !== undefined &&
    getStageSignature(getMessageStageKey(latestRenderableMessage)) !==
      getStageSignature(currentStageKey)

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`
    }
  }

  const saveProjectThumbnailFromImageBlock = useCallback(
    async (imageBlock: GeneratedImageBlock, overwrite = true) => {
      try {
        await saveGeneratedProjectThumbnail({
          supabase: createClient(),
          projectId,
          imageBlock,
          overwrite,
        })
      } catch (error) {
        console.error('[project thumbnail] failed to save generated image:', error)
      }
    },
    [projectId]
  )

  const insertMessage = useCallback(
    async ({
      role,
      content,
      activeAgent,
    }: {
      role: string
      content: string
      activeAgent?: string
    }) => {
      const supabase = createClient()
      const { data: lastMessage } = await supabase
        .from('messages')
        .select('seq_order')
        .eq('project_id', projectId)
        .order('seq_order', { ascending: false })
        .limit(1)
        .maybeSingle()

      const nextSeqOrder = (lastMessage?.seq_order ?? 0) + 1

      const { error } = await supabase.from('messages').insert({
        project_id: projectId,
        role,
        content,
        seq_order: nextSeqOrder,
        active_agent: activeAgent ?? 'aidee',
      })

      if (error) {
        throw error
      }
    },
    [projectId]
  )

  const fetchStageTimeline = useCallback(async () => {
    const supabase = createClient()
    const { data, error } = await supabase
      .from('design_stages')
      .select('stage_key, entered_at, exited_at, stage_order')
      .eq('project_id', projectId)
      .order('entered_at', { ascending: true })

    if (error) {
      console.error('[stage timeline] failed to fetch stages:', error)
      return
    }

    const timeline = ((data ?? []) as Array<{
      stage_key: string
      entered_at: string
      exited_at: string | null
      stage_order: number
    }>)
      .filter((stage) => isKnownStageKey(stage.stage_key))
      .map((stage) => ({
        ...stage,
        stage_key: stage.stage_key as StageKey,
      }))

    setStageTimeline(timeline)
  }, [projectId])

  const transitionStage = useCallback(
    async (nextStageKey: StageKey, exitReason = 'transition') => {
      if (!sessionId) {
        return
      }

      const response = await fetch('/api/study/stage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          projectId,
          nextStageKey,
          exitReason,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Stage transition failed: ${response.status} ${errorText}`)
      }

      const data = (await response.json()) as {
        currentStageKey?: StageKey
      }

      if (data.currentStageKey) {
        setCurrentStageKey(data.currentStageKey)
        await fetchStageTimeline()
      }
    },
    [fetchStageTimeline, projectId, sessionId]
  )

  const applyStageHeaders = useCallback(
    async (response: Response, aiContent: string) => {
      const currentStageHeader = response.headers.get('x-aidee-current-stage')
      const nextStageHeader = response.headers.get('x-aidee-next-stage')
      const transitionHeader = response.headers.get('x-aidee-transition')
      const reasonHeader = response.headers.get('x-aidee-reason') ?? 'transition'

      if (
        currentStageHeader &&
        isKnownStageKey(currentStageHeader) &&
        currentStageHeader !== currentStageKey
      ) {
        await transitionStage(currentStageHeader, `${reasonHeader}_resync`)
        return
      }

      if (
        transitionHeader === 'yes' &&
        nextStageHeader &&
        isKnownStageKey(nextStageHeader)
      ) {
        await transitionStage(nextStageHeader, reasonHeader)
        return
      }

      if (currentStageKey === 'step_1_idea' && parsePersonaData(aiContent)) {
        await transitionStage('step_2_persona', 'persona_generated')
      }
    },
    [currentStageKey, transitionStage]
  )

  useEffect(() => {
    const createSession = async () => {
      const response = await fetch('/api/study/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Session init failed: ${response.status} ${errorText}`)
      }

      const data = (await response.json()) as {
        sessionId?: string
        currentStageKey?: StageKey
      }

      if (data.sessionId) {
        setSessionId(data.sessionId)
      }

      if (data.currentStageKey) {
        setCurrentStageKey(data.currentStageKey)
      }

      await fetchStageTimeline()
    }

    createSession().catch((error) => console.error(error))
  }, [fetchStageTimeline, projectId])

  useEffect(() => {
    const fetchMessages = async () => {
      if (!projectId) {
        return
      }

      setIsInitialLoading(true)
      const supabase = createClient()
      const { data } = await supabase
        .from('messages')
        .select('id, role, content, seq_order, active_agent, created_at')
        .eq('project_id', projectId)
        .order('seq_order', { ascending: true })

      if (data) {
        const normalizedMessages = (data as ChatMessage[]).map((message) =>
          normalizeAssistantMessage(message)
        )

        const latestRfpFromHistory = [...normalizedMessages]
          .reverse()
          .find((message) => message.rfpJson)?.rfpJson
        const latestImageBlockFromHistory = [...normalizedMessages]
          .reverse()
          .find((message) => message.imageBlock)?.imageBlock

        if (latestRfpFromHistory) {
          setLatestRfpJson(latestRfpFromHistory)
        }

        if (latestImageBlockFromHistory) {
          setLatestGeneratedImageBlock(latestImageBlockFromHistory)
          void saveProjectThumbnailFromImageBlock(latestImageBlockFromHistory, false)
        }

        setMessages(
          normalizedMessages.map((message) => message.normalizedMessage)
        )
      }

      setIsInitialLoading(false)
    }

    fetchMessages()
  }, [projectId])

  useEffect(() => {
    const triggerInitialAI = async () => {
      if (
        searchParams.get('isNew') !== 'true' ||
        messages.length > 0 ||
        isLoading
      ) {
        return
      }

      setIsLoading(true)
      const supabase = createClient()
      const { data: project } = await supabase
        .from('projects')
        .select('requirements')
        .eq('id', projectId)
        .single()

      if (project?.requirements) {
        try {
          const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [],
              projectId,
              currentStageKey,
              activeExpert: 'aidee',
            }),
          })

          const reader = response.body?.getReader()
          if (!reader) {
            throw new Error('No response stream')
          }

          const decoder = new TextDecoder()
          let aiContent = ''
          const aiMessageId = Date.now().toString()
          const createdAt = new Date().toISOString()
          const responseStageKey = getResponseStageKey(response, currentStageKey)
          setMessages([
            {
              id: aiMessageId,
              role: 'assistant',
              content: '',
              active_agent: 'aidee',
              created_at: createdAt,
              stage_key: responseStageKey,
            },
          ])

          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              break
            }

            aiContent += decoder.decode(value, { stream: true })
            setMessages([
              {
                id: aiMessageId,
                role: 'assistant',
                content: aiContent,
                active_agent: 'aidee',
                created_at: createdAt,
                stage_key: responseStageKey,
              },
            ])
          }

          aiContent += decoder.decode()

          const { normalizedMessage, imageBlock, rfpJson } =
            normalizeAssistantMessage({
              id: aiMessageId,
              role: 'assistant',
              content: aiContent,
              active_agent: 'aidee',
              created_at: createdAt,
              stage_key: responseStageKey,
            })

          if (imageBlock) {
            setLatestGeneratedImageBlock(imageBlock)
            await saveProjectThumbnailFromImageBlock(imageBlock)
          }

          if (rfpJson) {
            setLatestRfpJson(rfpJson)
          }

          if (
            currentStageKey === 'step_5_rfp' ||
            currentStageKey === 'step_6_rfp' ||
            normalizedMessage.content.includes('# 제품 제안요청서') ||
            normalizedMessage.content.includes('## 1. 프로젝트 개요')
          ) {
            setLatestRfpContent(normalizedMessage.content)
          }

          setMessages([normalizedMessage])

          if (aiContent.trim()) {
            await insertMessage({
              role: 'assistant',
              content: aiContent,
              activeAgent: 'aidee',
            })
          }

          await applyStageHeaders(response, normalizedMessage.content)
        } catch (error) {
          console.error(error)
        }
      }
      setIsLoading(false)
    }

    if (!isInitialLoading) {
      triggerInitialAI()
    }
  }, [
    insertMessage,
    applyStageHeaders,
    saveProjectThumbnailFromImageBlock,
    projectId,
    isInitialLoading,
    isLoading,
    messages.length,
    searchParams,
    transitionStage,
    currentStageKey,
  ])

  useEffect(() => {
    const node = scrollRef.current
    if (node) {
      node.scrollTop = node.scrollHeight
    }
  }, [messages, isLoading])

  useEffect(() => {
    const latestRfpMessage = [...messages]
      .reverse()
      .find(
        (message) =>
          message.role === 'assistant' &&
          (message.content.includes('# 제품 제안요청서') ||
            message.content.includes('## 1. 프로젝트 개요'))
      )

    if (latestRfpMessage?.content) {
      setLatestRfpContent(latestRfpMessage.content)
    }
  }, [messages])

  useEffect(() => {
    const latestImageMessage = [...messages]
      .reverse()
      .find(
        (message) =>
          message.role === 'assistant' &&
          Array.isArray(message.generatedImages) &&
          message.generatedImages.length > 0
      )

    if (latestImageMessage?.generatedImages?.length) {
      setLatestGeneratedImageBlock({
        images: latestImageMessage.generatedImages,
        prompt: latestImageMessage.generatedImagePrompt ?? '',
        model: 'gemini-nanobanana',
      })
    }
  }, [messages])

  const handleRfpDownload = useCallback(async () => {
    try {
      setIsDownloadingRfp(true)

      const response = await fetch('/api/rfp/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId,
          rfpJson: latestRfpJson,
          rfpContent: latestRfpContent,
          projectTitle,
          layout: 'one-page',
        }),
      })

      if (!response.ok) {
        let errorMessage = `RFP download failed: ${response.status}`
        const rawText = await response.text()

        try {
          const parsed = JSON.parse(rawText) as { error?: string }
          errorMessage = parsed.error
            ? `RFP download failed: ${parsed.error}`
            : `${errorMessage} ${rawText}`
        } catch {
          errorMessage = `${errorMessage} ${rawText}`
        }

        throw new Error(errorMessage)
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${projectTitle || 'aidee-rfp'}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      console.error(error)
      alert(
        error instanceof Error
          ? error.message
          : 'RFP PDF를 생성하지 못했습니다.'
      )
    } finally {
      setIsDownloadingRfp(false)
    }
  }, [latestRfpContent, latestRfpJson, projectId, projectTitle])

  const streamAssistantResponse = async (
    nextMessages: ChatMessage[],
    stageKeyForRequest: StageKey = currentStageKey,
    expertForRequest: ExpertKey = activeExpert,
    expertCall = false
  ) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: buildChatApiMessages(nextMessages),
        projectId,
        currentStageKey: stageKeyForRequest,
        activeExpert: expertForRequest,
        expertCall,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Chat request failed: ${response.status} ${errorText}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const responseStageKey = getResponseStageKey(response, stageKeyForRequest)
    const decoder = new TextDecoder()
    let aiContent = ''
    const aiMessageId = (Date.now() + 1).toString()
    const createdAt = new Date().toISOString()

    setMessages((prev) => [
      ...prev,
      {
        id: aiMessageId,
        role: 'assistant',
        content: '',
        active_agent: expertForRequest,
        created_at: createdAt,
        stage_key: responseStageKey,
      },
    ])

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      aiContent += decoder.decode(value, { stream: true })
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId ? { ...msg, content: aiContent } : msg
        )
      )
    }

    aiContent += decoder.decode()

    const { normalizedMessage, imageBlock, rfpJson } = normalizeAssistantMessage({
      id: aiMessageId,
      role: 'assistant',
      content: aiContent,
      active_agent: expertForRequest,
      created_at: createdAt,
      stage_key: responseStageKey,
    })

    if (rfpJson) {
      setLatestRfpJson(rfpJson)
    }

    if (imageBlock) {
      setLatestGeneratedImageBlock(imageBlock)
      await saveProjectThumbnailFromImageBlock(imageBlock)
    }

    if (
      stageKeyForRequest === 'step_5_rfp' ||
      stageKeyForRequest === 'step_6_rfp' ||
      normalizedMessage.content.includes('# 제품 제안요청서') ||
      normalizedMessage.content.includes('## 1. 프로젝트 개요')
    ) {
      setLatestRfpContent(normalizedMessage.content)
    }

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === aiMessageId ? normalizedMessage : msg
      )
    )

    if (aiContent.trim()) {
      await insertMessage({
        role: 'assistant',
        content: aiContent,
        activeAgent: expertForRequest,
      })
    }

    await applyStageHeaders(response, normalizedMessage.content)
  }

  const selectExpert = (expert: ExpertKey) => {
    if (isLoading) {
      return
    }

    setActiveExpert(expert)
    setIsExpertPickerOpen(false)
  }

  const requestStageExpertAnswer = async (expert: ExpertKey) => {
    if (isLoading || expert === 'aidee') {
      return
    }

    setActiveExpert(expert)
    setIsExpertPickerOpen(false)
    setIsLoading(true)

    try {
      await streamAssistantResponse(messages, currentStageKey, expert, true)
    } catch (error) {
      console.error(error)
    } finally {
      setActiveExpert('aidee')
      setIsLoading(false)
    }
  }

  const requestNextStage = async () => {
    if (isLoading) {
      return
    }

    const nextStageKey = getNextStageKey(currentStageKey)
    if (!nextStageKey) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: '다음 단계로 진행해줘.',
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: nextStageKey,
    }

    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: userMessage.content,
        activeAgent: 'aidee',
      })

      await streamAssistantResponse(nextMessages, nextStageKey, 'aidee')
    } catch (error) {
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const getIntentStageKey = (text: string, fallbackStageKey: StageKey) => {
    if (/협력\s*업체|업체\s*연결|업체\s*추천|파트너|vendor|company/i.test(text)) {
      return 'step_6_company'
    }

    if (/rfp|제안요청서|제안\s*요청서|문서\s*생성|pdf/i.test(text)) {
      return 'step_6_rfp'
    }

    return fallbackStageKey
  }

  const onFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) {
      return
    }

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      active_agent: activeExpert,
      created_at: new Date().toISOString(),
      stage_key: getIntentStageKey(input, currentStageKey),
    }

    const nextMessages = [...messages, userMessage]
    const stageKeyForRequest = getIntentStageKey(input, currentStageKey)
    setMessages(nextMessages)
    setInput('')
    setIsExpertPickerOpen(false)
    setIsLoading(true)

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      await insertMessage({
        role: 'user',
        content: userMessage.content,
        activeAgent: activeExpert,
      })
      const isTemporaryExpertCall = activeExpert !== 'aidee'
      await streamAssistantResponse(
        nextMessages,
        stageKeyForRequest,
        activeExpert,
        isTemporaryExpertCall
      )
      if (isTemporaryExpertCall) {
        setActiveExpert('aidee')
      }
    } catch (error) {
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const sendPersonaAction = async (actionText: string) => {
    if (isLoading) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: actionText,
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }

    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: actionText,
        activeAgent: 'aidee',
      })

      let stageKeyForRequest = currentStageKey

      if (actionText === '리서치 진행') {
        await transitionStage('step_2_research', 'persona_confirmed')
        stageKeyForRequest = 'step_2_research'
      } else if (actionText === '페르소나 수정') {
        await transitionStage('step_2_persona', 'persona_revision_requested')
        stageKeyForRequest = 'step_2_persona'
      }

      await streamAssistantResponse(nextMessages, stageKeyForRequest, 'aidee')
    } catch (error) {
      console.error('Persona action failed:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content:
            '리서치 단계로 넘어가는 중 오류가 발생했어요. 다시 한 번 눌러주세요.',
          active_agent: 'aidee',
          created_at: new Date().toISOString(),
          stage_key: currentStageKey,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const sendGeneratedImageSelection = async (
    messageId: string,
    imageIndex: number,
    _prompt?: string | null
  ) => {
    if (isLoading) {
      return
    }

    const actionText = [
      `스타일 레퍼런스 ${imageIndex + 1}번을 선택합니다.`,
      '이 방향을 기준으로 형태, 색감, 재질 방향성을 정리하고 다음 단계로 진행해주세요.',
    ]
      .filter(Boolean)
      .join('\n')

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: actionText,
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }

    const nextMessages = [...messages, userMessage]

    setSelectedGeneratedImages((prev) => ({
      ...prev,
      [messageId]: imageIndex,
    }))
    setMessages(nextMessages)
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: actionText,
        activeAgent: 'aidee',
      })

      await streamAssistantResponse(nextMessages, currentStageKey, 'aidee')
    } catch (error) {
      console.error('Generated image selection failed:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content:
            '선택 결과를 반영하는 중 오류가 발생했어요. 다시 한 번 선택해주세요.',
          active_agent: 'aidee',
          created_at: new Date().toISOString(),
          stage_key: currentStageKey,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  if (isInitialLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-white">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
      </div>
    )
  }

  return (
    <div className="flex h-screen w-full overflow-hidden bg-white">
      <aside className="flex w-64 shrink-0 justify-center border-r border-gray-200 bg-neutral-50 p-2">
        <div className="flex h-full flex-1 flex-col justify-between">
          <div className="flex flex-col gap-8">
            <div className="flex w-60 items-center justify-between">
              <Link href="/" className="inline-block transition-opacity hover:opacity-80">
                <div className="h-5 w-14 rounded-full bg-gradient-to-r from-sky-500 to-blue-700" />
              </Link>
              <div className="h-5 w-24" />
              <form action="/auth/logout" method="post">
                <button
                  type="submit"
                  className="flex h-7 w-7 items-center justify-center rounded overflow-hidden"
                >
                  <div className="h-3.5 w-3.5 rounded-sm outline outline-[1.5px] outline-offset-[-0.75px] outline-gray-200" />
                </button>
              </form>
            </div>

            <div className="flex w-60 flex-col gap-1">
              <div className="text-xs leading-5 font-medium text-slate-500">
                디자인 프로세스
              </div>
              <div className="inline-flex items-start gap-1.5">
                <div className="inline-flex w-2.5 flex-col items-start">
                  {SIDEBAR_STEPS.map((_, index) => {
                    const currentIndex = getSidebarStepIndex(currentStageKey)
                    const isActive = index === currentIndex
                    const isLast = index === SIDEBAR_STEPS.length - 1

                    return (
                      <div
                        key={`dot-${index}`}
                        className="flex h-8 w-full flex-col items-center gap-px"
                      >
                        {index > 0 ? (
                          <div className="h-3 w-0 outline outline-1 outline-offset-[-0.50px] outline-gray-200" />
                        ) : null}
                        <div
                          className={`h-2.5 w-2.5 rounded-full ${
                            isActive ? 'border-2 border-blue-600 bg-white' : 'bg-gray-200'
                          }`}
                        />
                        {!isLast ? (
                          <div className="h-3 w-0 outline outline-1 outline-offset-[-0.50px] outline-gray-200" />
                        ) : null}
                      </div>
                    )
                  })}
                </div>
                <div className="inline-flex w-56 flex-col items-start">
                  {SIDEBAR_STEPS.map((step, index) => {
                    const isActive = index === getSidebarStepIndex(currentStageKey)
                    return (
                      <div
                        key={step}
                        className="inline-flex self-stretch items-center gap-2 py-1.5"
                      >
                        <div
                          className={`text-sm leading-6 font-medium ${
                            isActive ? 'text-blue-600' : 'text-gray-300'
                          }`}
                        >
                          {step}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="flex w-60 flex-col gap-2.5 border-t border-gray-200 px-2 pt-3 pb-1">
            <div className="inline-flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-5 w-5 rounded-full bg-gray-200" />
                <div className="text-sm leading-5 font-medium text-slate-500">
                  {userName}
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-[999px] bg-gradient-to-bl from-blue-600/0 to-blue-600/40 px-4 py-1">
                <div className="h-4 w-4 overflow-hidden">
                  <div className="h-3.5 w-3.5 outline outline-[1.5px] outline-offset-[-0.75px] outline-white" />
                </div>
                <div className="text-sm leading-5 font-medium text-white">
                  Basic
                </div>
              </div>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col items-center bg-white">
        <header className="flex h-16 w-full shrink-0 items-center justify-center bg-white">
          <div className="inline-flex gap-1 rounded-full bg-gray-100 p-1 shadow-inner">
            <button
              type="button"
              className="rounded-full bg-white px-8 py-1.5 text-sm font-semibold text-blue-600 shadow-sm"
            >
              채팅
            </button>
            <button
              type="button"
              className="px-8 py-1.5 text-sm font-medium text-gray-400 transition-colors hover:text-gray-600"
            >
              라이브러리
            </button>
          </div>
        </header>

        <div
          ref={scrollRef}
          className="scrollbar-hide w-full max-w-4xl flex-1 space-y-6 overflow-y-auto px-6 py-8"
        >
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <p className="text-sm font-medium text-sky-700">{projectTitle}</p>
              <p className="text-xs text-zinc-400">현재 단계: {currentStageKey}</p>
              {latestGeneratedImageBlock?.images.length ? (
                <p className="text-xs text-zinc-400">
                  최근 생성 이미지: {latestGeneratedImageBlock.images.length}장
                </p>
              ) : null}
            </div>
          </div>

          {messages.length === 0 && !isLoading ? (
            <>
              <StageDivider stageKey={currentStageKey} />
              <div className="max-w-[514px] rounded-[24px] rounded-tl-none bg-gray-200 p-5 text-base leading-relaxed font-medium text-neutral-900">
                안녕하세요! Aidee입니다. 기획 중인 프로젝트를 함께 정리해볼게요.
              </div>
            </>
          ) : null}

          {messages.map((m, index) => {
            if (nestedExpertMessageIds.has(m.id)) {
              return null
            }

            const stageKeyForMessage = getMessageStageKey(m)
            const stageSignature = getStageSignature(stageKeyForMessage)
            const previousRenderableMessage = messages
              .slice(0, index)
              .reverse()
              .find((message) => !nestedExpertMessageIds.has(message.id))
            const previousStageSignature = previousRenderableMessage
              ? getStageSignature(getMessageStageKey(previousRenderableMessage))
              : null
            const shouldShowStageDivider =
              stageSignature !== previousStageSignature

            const isPersonaCard =
              m.role === 'assistant' &&
              (m.content.includes('Persona Card') ||
                (m.content.includes('User') &&
                  m.content.includes('Problem') &&
                  m.content.includes('Decision')))

            const isRfpMessage =
              m.role === 'assistant' &&
              (m.content.includes('# 제품 제안요청서') ||
                m.content.includes('## 1. 프로젝트 개요'))
            const canShowAdvanceButton =
              m.role === 'assistant' &&
              m.id === latestAideeAssistantId &&
              ['step_2_research', 'step_3_direction', 'step_5_design'].includes(
                currentStageKey
              ) &&
              Boolean(getNextStageKey(currentStageKey))

            if (isPersonaCard) {
              const personaData = parsePersonaData(m.content)

              if (personaData) {
                return (
                  <div key={m.id} className="space-y-6">
                    {shouldShowStageDivider ? (
                      <StageDivider stageKey={stageKeyForMessage} />
                    ) : null}
                    <PersonaCard
                      data={personaData}
                      onProceed={() => sendPersonaAction('리서치 진행')}
                      onAdjust={() => sendPersonaAction('페르소나 수정')}
                    />
                  </div>
                )
              }
            }

            return (
              <div
                key={m.id}
                className="space-y-6"
              >
                {shouldShowStageDivider ? (
                  <StageDivider stageKey={stageKeyForMessage} />
                ) : null}
                <div
                  className={`flex flex-col ${
                    m.role === 'user' ? 'items-end' : 'items-start'
                  }`}
                >
                  {m.role === 'assistant' && isExpertKey(m.active_agent) ? (
                    <div className="mb-1 flex items-center gap-1.5 px-1">
                      <ExpertAvatar
                        expertKey={m.active_agent}
                        selected
                        className="h-5 w-5"
                      />
                      <span className="text-xs font-medium text-slate-400">
                        {getExpertDefinition(m.active_agent).label}
                      </span>
                    </div>
                  ) : null}
                  <div
                    className={`max-w-[514px] rounded-[24px] p-5 text-base leading-relaxed font-medium shadow-sm ${
                      m.role === 'user'
                        ? 'rounded-tr-none bg-gray-100 text-neutral-900'
                        : 'rounded-tl-none bg-gray-200 text-neutral-900'
                    }`}
                  >
                    <div className="prose prose-sm prose-p:my-0 prose-p:leading-7 prose-li:my-0 prose-headings:mb-3 prose-strong:text-neutral-900 max-w-none break-words whitespace-pre-wrap">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkBreaks]}
                        components={{
                          p: ({ children }) => (
                            <p className="mb-1 leading-7 last:mb-0">
                              {children}
                            </p>
                          ),
                          ul: ({ children }) => (
                            <ul className="my-3 list-disc space-y-1 pl-5">
                              {children}
                            </ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="my-3 list-decimal space-y-1 pl-5">
                              {children}
                            </ol>
                          ),
                          li: ({ children }) => (
                            <li className="leading-5 [&>p]:mb-0 [&>p]:inline">
                              {children}
                            </li>
                          ),
                          br: () => <br className="block h-1" />,
                        }}
                      >
                        {m.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                {m.role === 'assistant' && m.generatedImages?.length ? (
                  <div className="mt-3 w-full max-w-[760px] space-y-2">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-medium text-slate-500">
                        생성된 이미지 {m.generatedImages.length}장
                      </p>
                      {m.generatedImagePrompt ? (
                        <p className="max-w-[440px] truncate text-right text-[11px] text-slate-400">
                          {m.generatedImagePrompt}
                        </p>
                      ) : null}
                    </div>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {m.generatedImages.map((image, index) => (
                        <div
                          key={`${m.id}-image-${index}`}
                          className={`overflow-hidden rounded-2xl border bg-white ${
                            selectedGeneratedImages[m.id] === index
                              ? 'border-blue-600 ring-2 ring-blue-100'
                              : 'border-slate-200'
                          }`}
                        >
                          <img
                            src={image}
                            alt={`generated-${index + 1}`}
                            className="aspect-[4/3] w-full object-cover"
                          />
                          {currentStageKey === 'step_4_style' ? (
                            <div className="flex items-center justify-between gap-3 border-t border-slate-100 px-3 py-2">
                              <span className="text-xs font-medium text-slate-500">
                                레퍼런스 {index + 1}
                              </span>
                              <button
                                type="button"
                                disabled={
                                  isLoading ||
                                  selectedGeneratedImages[m.id] !== undefined
                                }
                                onClick={() =>
                                  void sendGeneratedImageSelection(
                                    m.id,
                                    index,
                                    m.generatedImagePrompt
                                  )
                                }
                                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                                  selectedGeneratedImages[m.id] === index
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-slate-100 text-slate-700 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-60'
                                }`}
                              >
                                {selectedGeneratedImages[m.id] === index
                                  ? '선택됨'
                                  : '이 이미지 선택'}
                              </button>
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
                {isRfpMessage ? (
                  <RfpActionPanel
                    isDownloadingRfp={isDownloadingRfp}
                    isLoading={isLoading}
                    onDownload={() => void handleRfpDownload()}
                    onCompanyConnect={() =>
                      setInput('협력업체 연결을 진행해주세요.')
                    }
                  />
                ) : null}
                {canShowAdvanceButton ? (
                  <div className="mt-3 flex items-center gap-2">
                    <StageAdvanceButton
                      label="다음 단계로 넘어갈까요?"
                      onClick={() => void requestNextStage()}
                      disabled={isLoading}
                    />
                  </div>
                ) : null}
                {m.role === 'assistant' &&
                (!m.active_agent || m.active_agent === 'aidee') &&
                expertRepliesByAideeId.get(m.id)?.length ? (
                  <div className="mt-2 flex w-full max-w-[602px] flex-col gap-2">
                    {expertRepliesByAideeId.get(m.id)?.map((reply) => {
                      const expertKey = isExpertKey(reply.active_agent)
                        ? reply.active_agent
                        : 'aidee'
                      const definition = getExpertDefinition(expertKey)
                      const isReplyLoading = !reply.content.trim()

                      return (
                        <div
                          key={reply.id}
                          className="ml-5 rounded-2xl bg-gray-100 p-3 text-sm font-medium leading-6 text-neutral-800"
                        >
                          <div className="mb-1.5 flex items-center gap-1.5">
                            <ExpertAvatar
                              expertKey={expertKey}
                              selected
                              className="h-5 w-5"
                            />
                            <span className="text-xs font-medium text-slate-400">
                              {definition.label}
                            </span>
                          </div>
                          {isReplyLoading ? (
                            <div className="flex min-h-16 flex-col items-center justify-center gap-2 text-slate-400">
                              <div className="h-6 w-6 animate-spin rounded-full border-4 border-slate-300 border-t-transparent" />
                              <span>{definition.loadingLabel}</span>
                            </div>
                          ) : (
                            <div className="prose prose-sm prose-p:my-0 prose-p:leading-6 prose-li:my-0 max-w-none break-words whitespace-pre-wrap">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkBreaks]}
                                components={{
                                  p: ({ children }) => (
                                    <p className="mb-1 leading-6 last:mb-0">
                                      {children}
                                    </p>
                                  ),
                                  ul: ({ children }) => (
                                    <ul className="my-2 list-disc space-y-1 pl-5">
                                      {children}
                                    </ul>
                                  ),
                                  ol: ({ children }) => (
                                    <ol className="my-2 list-decimal space-y-1 pl-5">
                                      {children}
                                    </ol>
                                  ),
                                  li: ({ children }) => (
                                    <li className="leading-5 [&>p]:mb-0 [&>p]:inline">
                                      {children}
                                    </li>
                                  ),
                                  br: () => <br className="block h-1" />,
                                }}
                              >
                                {reply.content}
                              </ReactMarkdown>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                ) : null}
                {m.role === 'assistant' &&
                m.id === latestAideeAssistantId &&
                (!m.active_agent || m.active_agent === 'aidee') &&
                m.content.trim() &&
                expertSuggestionCandidates.length > 0 ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-1.5">
                    {expertSuggestionCandidates.map((expert) => {
                      const definition = getExpertDefinition(expert.key)

                      return (
                        <button
                          key={`${m.id}-${expert.key}`}
                          type="button"
                          disabled={isLoading}
                          onClick={() => void requestStageExpertAnswer(expert.key)}
                          className="inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-slate-500 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 hover:text-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <ExpertAvatar
                            expertKey={expert.key}
                            selected
                            className="h-5 w-5"
                          />
                          {definition.label} 답변 생성
                        </button>
                      )
                    })}
                  </div>
                ) : null}
                </div>
              </div>
            )
          })}

          {shouldShowCurrentStageDivider ? (
            <StageDivider stageKey={currentStageKey} />
          ) : null}

          {currentStageKey === 'step_6_company' && latestRfpContent ? (
            <CompanyRecommendationsPanel
              rfpJson={latestRfpJson}
              rfpContent={latestRfpContent}
            />
          ) : null}

          {isLoading && activeExpert === 'aidee' ? (
            <div className="flex items-center gap-3 rounded-3xl bg-gray-100 px-5 py-4">
              <div className="flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400" />
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:0.2s]" />
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:0.4s]" />
              </div>
              <p className="text-sm font-medium text-slate-500">
                {activeExpertDefinition.loadingLabel}
              </p>
            </div>
          ) : null}
        </div>

        <footer className="w-full max-w-4xl p-6 pb-10">
          <form onSubmit={onFormSubmit} className="group relative">
            {isExpertPickerOpen ? (
              <div className="absolute bottom-[68px] left-0 z-20 w-52 overflow-hidden rounded-[20px] bg-white px-1 py-2.5 shadow-[0px_16px_60px_0px_rgba(0,0,0,0.08)] outline outline-1 outline-gray-100">
                <div className="px-4 pb-2 pt-1.5 text-xs font-semibold leading-5 text-slate-400">
                  AI 전문가 선택
                </div>
                <div className="flex flex-col gap-1">
                  {selectableExperts.map((expert) => {
                    const isActive = expert.key === activeExpert

                    return (
                      <button
                        key={expert.key}
                        type="button"
                        disabled={isLoading}
                        onClick={() => selectExpert(expert.key)}
                        className={`flex w-full items-center gap-2 rounded-2xl px-3 py-2 text-left transition disabled:cursor-not-allowed disabled:opacity-50 ${
                          isActive
                            ? 'bg-gray-100 text-neutral-900'
                            : 'text-slate-500 hover:bg-gray-50 hover:text-neutral-900'
                        }`}
                      >
                        <ExpertAvatar expertKey={expert.key} selected={isActive} />
                        <span className="text-sm font-medium leading-5">
                          {expert.label}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            ) : null}

            <div className="flex min-h-[56px] items-end gap-3 rounded-[20px] bg-white p-2 shadow-[0px_4px_24px_0px_rgba(0,0,0,0.08)] outline outline-1 outline-gray-200 transition-all focus-within:outline-blue-200">
              <button
                type="button"
                disabled={isLoading}
                onClick={() => setIsExpertPickerOpen((prev) => !prev)}
                className="mb-0.5 flex h-10 min-w-10 items-center justify-center rounded-full transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                title="AI 전문가 선택"
              >
                {activeExpert === 'aidee' ? (
                  <span className="relative h-8 w-8 overflow-hidden rounded-full bg-gray-100">
                    <span className="absolute left-[7px] top-[7px] h-4 w-4 rounded-full bg-gray-300" />
                    <span className="absolute left-[12px] top-[12px] h-1.5 w-1.5 rounded-full bg-gray-100" />
                  </span>
                ) : (
                  <ExpertAvatar expertKey={activeExpert} selected />
                )}
              </button>
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={handleInput}
                onFocus={() => setIsExpertPickerOpen(false)}
                onKeyDown={(e) => {
                  const nativeEvent = e.nativeEvent as KeyboardEvent & {
                    isComposing?: boolean
                  }

                  if (
                    e.key === 'Enter' &&
                    !e.shiftKey &&
                    !nativeEvent.isComposing &&
                    nativeEvent.keyCode !== 229
                  ) {
                    e.preventDefault()
                    void onFormSubmit(e)
                  }
                }}
                placeholder={
                  isExpertPickerOpen
                    ? '원하는 전문가를 선택해서 물어보세요.'
                    : activeExpertDefinition.inputLabel
                }
                className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-3 text-base leading-relaxed font-medium outline-none placeholder:text-zinc-400"
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className={`mb-0.5 flex h-10 w-10 items-center justify-center rounded-full transition-all ${
                  input.trim()
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-300'
                }`}
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="m5 12 7-7 7 7" />
                  <path d="M12 19V5" />
                </svg>
              </button>
            </div>
          </form>
        </footer>
      </main>
    </div>
  )
}
