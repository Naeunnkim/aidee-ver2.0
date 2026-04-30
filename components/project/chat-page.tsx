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
import { createClient } from '@/lib/supabase/client'
import {
  SIDEBAR_STEPS,
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
  generatedImages?: string[]
  generatedImagePrompt?: string | null
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

function parsePersonaData(content: string) {
  const normalizedContent = content.replace(/\r\n/g, '\n')

  const imageUrlMatch = normalizedContent.match(
    /https?:\/\/[^\s]+(?:\.jpg|\.jpeg|\.png|\.webp|\/seed\/[^\s]+)/i
  )
  const imageUrl = imageUrlMatch ? imageUrlMatch[0] : ''

  const sectionTitles = [
    'User',
    'Usage',
    'Problem',
    'Current Solution',
    'Decision',
    'Success',
  ]

  const extractSection = (title: string) => {
    const nextTitles = sectionTitles.filter((item) => item !== title).join('|')
    const regex = new RegExp(
      `(?:^|\\n)(?:##+\\s*|\\*\\*\\s*)?${title}(?:\\s*\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n(?:##+\\s*|\\*\\*\\s*)?(?:${nextTitles})(?:\\s*\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalizedContent.match(regex)
    if (!match) {
      return []
    }

    return match[1]
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .filter(
        (line) =>
          line.startsWith('-') || line.startsWith('•') || line.startsWith('#')
      )
      .map((line) => line.replace(/^[-•]\s*/, '').trim())
      .filter((line) => line.length > 0)
      .filter((line) => !/^[-–—]+$/.test(line))
  }

  const successLines = extractSection('Success')
  const successData = successLines.map((line) => {
    const normalized = line.replace(/^#/, '').trim()
    const [tag, ...descParts] = normalized.split('→').map((s) => s.trim())
    return {
      tag: tag || normalized,
      desc: descParts.join(' ') || '',
    }
  })

  const parsed = {
    user: extractSection('User'),
    usage: extractSection('Usage'),
    problem: extractSection('Problem'),
    currentSolution: extractSection('Current Solution'),
    decision: extractSection('Decision'),
    success: successData,
    imageUrl,
  }

  const hasMinimumData =
    parsed.user.length > 0 ||
    parsed.usage.length > 0 ||
    parsed.problem.length > 0 ||
    parsed.currentSolution.length > 0 ||
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
      }
    },
    [projectId, sessionId]
  )

  const applyStageHeaders = useCallback(
    async (response: Response, aiContent: string) => {
      const nextStageHeader = response.headers.get('x-aidee-next-stage')
      const transitionHeader = response.headers.get('x-aidee-transition')
      const reasonHeader = response.headers.get('x-aidee-reason') ?? 'transition'

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
    }

    createSession().catch((error) => console.error(error))
  }, [projectId])

  useEffect(() => {
    const fetchMessages = async () => {
      if (!projectId) {
        return
      }

      setIsInitialLoading(true)
      const supabase = createClient()
      const { data } = await supabase
        .from('messages')
        .select('id, role, content, seq_order, active_agent')
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
        }

        const latestExpertFromHistory = [...normalizedMessages]
          .reverse()
          .find((message) =>
            isExpertKey(message.normalizedMessage.active_agent)
          )?.normalizedMessage.active_agent

        if (isExpertKey(latestExpertFromHistory)) {
          setActiveExpert(latestExpertFromHistory)
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
          setMessages([{ id: aiMessageId, role: 'assistant', content: '' }])

          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              break
            }

            aiContent += decoder.decode(value, { stream: true })
            setMessages([
              { id: aiMessageId, role: 'assistant', content: aiContent },
            ])
          }

          aiContent += decoder.decode()

          const { normalizedMessage, imageBlock, rfpJson } =
            normalizeAssistantMessage({
              id: aiMessageId,
              role: 'assistant',
              content: aiContent,
            })

          if (imageBlock) {
            setLatestGeneratedImageBlock(imageBlock)
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
        messages: nextMessages,
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

    const decoder = new TextDecoder()
    let aiContent = ''
    const aiMessageId = (Date.now() + 1).toString()

    setMessages((prev) => [
      ...prev,
      {
        id: aiMessageId,
        role: 'assistant',
        content: '',
        active_agent: expertForRequest,
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
    })

    if (rfpJson) {
      setLatestRfpJson(rfpJson)
    }

    if (imageBlock) {
      setLatestGeneratedImageBlock(imageBlock)
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
    }

    const nextMessages = [...messages, userMessage]
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
      await streamAssistantResponse(nextMessages, currentStageKey, activeExpert)
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
      active_agent: activeExpert,
    }

    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: actionText,
        activeAgent: activeExpert,
      })

      let stageKeyForRequest = currentStageKey

      if (actionText === '리서치 진행') {
        await transitionStage('step_2_research', 'persona_confirmed')
        stageKeyForRequest = 'step_2_research'
      } else if (actionText === '페르소나 수정') {
        await transitionStage('step_2_persona', 'persona_revision_requested')
        stageKeyForRequest = 'step_2_persona'
      }

      await streamAssistantResponse(nextMessages, stageKeyForRequest, activeExpert)
    } catch (error) {
      console.error('Persona action failed:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content:
            '리서치 단계로 넘어가는 중 오류가 발생했어요. 다시 한 번 눌러주세요.',
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const sendGeneratedImageSelection = async (
    messageId: string,
    imageIndex: number,
    prompt?: string | null
  ) => {
    if (isLoading) {
      return
    }

    const actionText = [
      `스타일 레퍼런스 ${imageIndex + 1}번을 선택합니다.`,
      prompt ? `선택 기준 프롬프트: ${prompt}` : '',
      '이 방향을 기준으로 다음 디자인 제안을 진행해주세요.',
    ]
      .filter(Boolean)
      .join('\n')

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: actionText,
      active_agent: activeExpert,
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
        activeAgent: activeExpert,
      })

      await streamAssistantResponse(nextMessages, currentStageKey, activeExpert)
    } catch (error) {
      console.error('Generated image selection failed:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content:
            '선택 결과를 반영하는 중 오류가 발생했어요. 다시 한 번 선택해주세요.',
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
            {currentStageKey === 'step_6_rfp' ||
            currentStageKey === 'step_5_rfp' ||
            latestRfpContent ? (
              <button
                type="button"
                onClick={() => void handleRfpDownload()}
                disabled={isDownloadingRfp || isLoading}
                className="rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isDownloadingRfp ? 'RFP 생성 중...' : 'RFP 다운로드'}
              </button>
            ) : null}
          </div>

          <div className="flex items-center gap-2">
            <div className="h-px flex-1 bg-gray-100" />
            <ExpertAvatar expertKey={activeExpert} selected />
            <p className="text-xs font-medium text-slate-500">
              {activeExpertDefinition.label}가 함께하고 있어요.
            </p>
            <div className="h-px flex-1 bg-gray-100" />
          </div>

          {messages.length === 0 && !isLoading ? (
            <div className="max-w-[514px] rounded-[24px] rounded-tl-none bg-gray-200 p-5 text-base leading-relaxed font-medium text-neutral-900">
              안녕하세요! Aidee입니다. 기획 중인 프로젝트를 함께 정리해볼게요.
            </div>
          ) : null}

          {messages.map((m) => {
            const isPersonaCard =
              m.role === 'assistant' &&
              (m.content.includes('Persona Card') ||
                (m.content.includes('User') &&
                  m.content.includes('Problem') &&
                  m.content.includes('Decision')))

            if (isPersonaCard) {
              const personaData = parsePersonaData(m.content)

              if (personaData) {
                return (
                  <PersonaCard
                    key={m.id}
                    data={personaData}
                    onProceed={() => sendPersonaAction('리서치 진행')}
                    onAdjust={() => sendPersonaAction('페르소나 수정')}
                  />
                )
              }
            }

            return (
              <div
                key={m.id}
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
                          <p className="mb-1 leading-7 last:mb-0">{children}</p>
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
              </div>
            )
          })}

          {isLoading ? (
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
                  if (e.key === 'Enter' && !e.shiftKey) {
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
