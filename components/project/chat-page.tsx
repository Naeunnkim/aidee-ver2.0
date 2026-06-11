'use client'

import Image from 'next/image'
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
  PROCESS_STEPS,
  SIDEBAR_STEPS,
  STAGE_DEFINITIONS,
  getNextStageKey,
  getProcessStepForStage,
  getSidebarStepIndex,
  getStageKeysForSidebarIndex,
  isKnownStageKey,
  isSameOrNextStage,
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
  generatedImagePurpose?: GeneratedImageBlock['purpose'] | null
}

type ChatApiMessage = {
  role: string
  content: string
}

type ChatChoice = {
  key: 'A' | 'B' | 'C'
  label: string
  value: string
}

type ForceImageGeneration =
  | 'initial_design'
  | 'design_revision'
  | 'problem_statements_visualization'
  | 'experience_keywords_visualization'
  | 'relationship_keywords_visualization'
  | 'market_size_visualization'
  | 'consumption_keywords_visualization'
  | 'brand_positioning_visualization'
  | 'style_reference_options'
  | 'style_moodboard_visualization'
  | 'persona_visualization'

type PersonaArtifactKind =
  | 'problem_statements'
  | 'experience_keywords'
  | 'relationship_keywords'
  | 'persona'

type DirectionArtifactKind =
  | 'market_size'
  | 'consumption_keywords'
  | 'brand_positioning'

type ProjectDirectionData = {
  title: string
  goal: string
  category: string
  budgetMinimum: string
  budgetRange: string
  duration: string
  budgetAndDuration: string
  size: string
  features: string
  usage: string
  ideaSummary: string
  referenceSummary: string
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
  thumbnail: string
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
      generatedImagePurpose: imageBlock?.purpose ?? null,
    } satisfies ChatMessage,
    imageBlock,
    rfpJson,
  }
}

function buildChatApiMessages(messages: ChatMessage[]): ChatApiMessage[] {
  return messages.map((message) => {
    const { cleanedText, imageBlock } = extractGeneratedImagesBlock(
      message.content
    )
    const generatedImagePurpose =
      imageBlock?.purpose ?? message.generatedImagePurpose
    const generatedImageCount =
      imageBlock?.images.length ?? message.generatedImages?.length ?? 0
    const imageContext =
      message.role === 'assistant' && generatedImagePurpose && generatedImageCount
        ? [
            '',
            `[시스템 참고: 이 assistant 응답에는 ${generatedImagePurpose} 이미지 ${generatedImageCount}장이 생성되어 있었습니다.]`,
            generatedImagePurpose === 'design' && generatedImageCount >= 4
              ? '[시스템 참고: STEP5 초기 디자인 4안 세트가 이미 제시되었습니다. 이후에는 같은 4안 세트를 반복 생성하지 말고 사용자가 선택한 1안을 발전시키거나 확정 처리하세요.]'
              : '',
          ]
            .filter(Boolean)
            .join('\n')
        : ''

    return {
      role: message.role,
      content: `${cleanedText}${imageContext}`.trim(),
    }
  })
}

function stripInternalBlocksForDisplay(text: string) {
  return text
    .replace(
      /\n?<<\s*AIDEE[-_ ]?(?:IMAGES|RFP_JSON)\s*>>[\s\S]*?(?:<<\s*\/\s*AIDEE[-_ ]?(?:IMAGES|RFP_JSON)\s*>>|$)/gi,
      ''
    )
    .replace(
      /\n?<<AIDEE_PERSONA_FLOW_CARD:[\s\S]*?<<\/AIDEE_PERSONA_FLOW_CARD>>/g,
      ''
    )
    .replace(
      /\n?<<AIDEE_DIRECTION_WIDGETS>>[\s\S]*?<<\/AIDEE_DIRECTION_WIDGETS>>/g,
      ''
    )
    .replace(
      /\n?<<AIDEE_DIRECTION_CARD:[\s\S]*?<<\/AIDEE_DIRECTION_CARD>>/g,
      ''
    )
    .replace(
      /\n?<<AIDEE_STYLE_KEYWORD_PICKER>>[\s\S]*?<<\/AIDEE_STYLE_KEYWORD_PICKER>>/g,
      ''
    )
    .replace(
      /\n?<<AIDEE_PROJECT_DIRECTION>>[\s\S]*?<<\/AIDEE_PROJECT_DIRECTION>>/g,
      ''
    )
    .replace(
      /\n?#\s*Project\s*(?:Card|Direction)\s*[\s\S]*?(?=\n\s*제품의\s*구체적인\s*모습|\n\s*형태,\s*색감|\s*$)/gi,
      ''
    )
    .replace(
      /\n?(?:수정 내용을 반영했습니다\.\s*)?내용을 확인한 뒤 확정하기를 누르면 시각화하기 버튼이 나타납니다\./g,
      ''
    )
    .replace(/\n?\[시스템\s*참고:[\s\S]*?\]/gi, '')
    .trim()
}

function splitAssistantChoices(content: string): {
  displayContent: string
  choices: ChatChoice[]
} {
  const lines = content.replace(/\r\n/g, '\n').split('\n')
  const choiceLinePattern = /^(?:[-•*]\s*)?([ABC])\.\s+(.+)$/
  const nonChoiceLabels = [
    /^User$/i,
    /^Behavior Map$/i,
    /^Correlation Analysis$/i,
    /^Problem$/i,
    /^Success$/i,
    /^Decision$/i,
    /^프로젝트 개요$/,
    /^페르소나$/,
    /^타겟 리서치$/,
  ]
  const removableLineIndexes = new Set<number>()
  let choices: ChatChoice[] = []
  let candidateGroup: Array<ChatChoice & { lineIndex: number }> = []

  const flushCandidateGroup = () => {
    const keys = candidateGroup.map((choice) => choice.key).join('')
    const hasInteractiveChoiceGroup =
      candidateGroup.length === 3 && keys === 'ABC'

    if (hasInteractiveChoiceGroup) {
      choices = [
        ...choices,
        ...candidateGroup.map((choice) => ({
          key: choice.key,
          label: choice.label,
          value: choice.value,
        })),
      ]
      candidateGroup.forEach((choice) => removableLineIndexes.add(choice.lineIndex))

      const firstChoiceLineIndex = candidateGroup[0]?.lineIndex
      const previousLineIndex =
        typeof firstChoiceLineIndex === 'number' ? firstChoiceLineIndex - 1 : -1

      if (/^\s*선택지\s*:?\s*$/.test(lines[previousLineIndex] ?? '')) {
        removableLineIndexes.add(previousLineIndex)
      }
    }

    candidateGroup = []
  }

  lines.forEach((line, index) => {
    const trimmedLine = line.trim()
    const inlineChoiceMatch = trimmedLine.match(
      /^A\.\s*(.+?)\s*\/\s*B\.\s*(.+?)\s*\/\s*C\.\s*(.+)$/
    )
    const match = trimmedLine.match(choiceLinePattern)

    if (inlineChoiceMatch) {
      flushCandidateGroup()
      const inlineChoices = inlineChoiceMatch.slice(1, 4).map((label, labelIndex) => {
        const key = ['A', 'B', 'C'][labelIndex] as ChatChoice['key']
        const normalizedLabel = label.trim()

        return {
          key,
          label: normalizedLabel,
          value: `${key}. ${normalizedLabel}`,
        }
      })

      choices = [...choices, ...inlineChoices]
      removableLineIndexes.add(index)
      return
    }

    if (!match) {
      if (!trimmedLine && candidateGroup.length > 0) {
        return
      }

      flushCandidateGroup()
      return
    }

    const key = match[1] as ChatChoice['key']
    const label = match[2].trim()
    const isSectionLabel = nonChoiceLabels.some((pattern) => pattern.test(label))

    if (isSectionLabel) {
      flushCandidateGroup()
      return
    }

    candidateGroup.push({
      key,
      label,
      value: `${key}. ${label}`,
      lineIndex: index,
    })
  })
  flushCandidateGroup()

  if (choices.length === 0) {
    return {
      displayContent: content,
      choices: [],
    }
  }

  const displayContent = lines
    .filter((_, index) => !removableLineIndexes.has(index))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

  return {
    displayContent,
    choices,
  }
}

function isStageProceedPrompt(content: string) {
  const normalized = content.replace(/\s+/g, ' ').trim()
  const mentionsProceedQuestion = /진행할까요[?？]?/.test(normalized)
  const mentionsNextStage =
    /다음(?:으로| 단계| STEP)/i.test(normalized) ||
    /STEP\s*\d+[\s\S]*(?:넘어|진행|이동|들어가|시작)/i.test(normalized)

  return mentionsProceedQuestion && mentionsNextStage
}

function getStageKeyFromProceedPrompt(content: string): StageKey | null {
  const normalized = content.replace(/\s+/g, ' ')
  const stepPatterns = [
    /다음(?:으로)?\s*STEP\s*(\d+)/i,
    /다음\s*단계(?:로|으로)?[\s\S]*?STEP\s*(\d+)/i,
    /STEP\s*(\d+)[\s\S]*?(?:넘어|진행|이동|들어가|시작)/i,
  ]
  const match = stepPatterns
    .map((pattern) => normalized.match(pattern))
    .find(Boolean)
  const stepIndex = match ? Number(match[1]) : Number.NaN

  if (Number.isNaN(stepIndex)) {
    return null
  }

  return (
    PROCESS_STEPS.find((step) => step.index === stepIndex)?.stageKeys[0] ??
    null
  )
}

function splitBudgetAndDuration(value: string) {
  const [budgetRange = '', duration = ''] = value.split('/').map((item) => item.trim())

  return { budgetRange, duration }
}

function extractProjectDirectionCard(content: string): ProjectDirectionData | null {
  const markerMatch = content.match(
    /<<AIDEE_PROJECT_DIRECTION>>\s*([\s\S]*?)\s*<<\/AIDEE_PROJECT_DIRECTION>>/
  )
  const headingMatch =
    markerMatch ??
    content.match(
      /#\s*Project\s*(?:Card|Direction)\s*([\s\S]*?)(?=\n\s*제품의\s*구체적인\s*모습|\n\s*형태,\s*색감|\s*$)/i
    )
  const source = headingMatch?.[1]

  if (!source) {
    return null
  }

  const fieldLabels = [
    '프로젝트명',
    '프로젝트 목표',
    '제품 카테고리',
    '예산/기간 범위',
    'Budget Minimum',
    'Target Timeline',
    'Project Scope',
    '예상 크기',
    '주요 기능',
    'Key Features',
    '최종 활용 목적',
    '아이디어 정리',
    '참고 자료',
  ]
  const labelSet = new Set(fieldLabels)
  const values = new Map<string, string>()
  const cleanLine = (line: string) =>
    line
      .replace(/^#{1,6}\s*/, '')
      .replace(/\*\*/g, '')
      .replace(/^[-•]\s*/, '')
      .replace(/\s+/g, ' ')
      .trim()
  const lines = source
    .replace(/^Project\s*Direction\s*/i, '')
    .split('\n')
    .map(cleanLine)
    .filter(Boolean)

  let currentLabel: string | null = null

  for (const line of lines) {
    if (labelSet.has(line)) {
      currentLabel = line
      values.set(currentLabel, '')
      continue
    }

    if (!currentLabel || /^Project\s*Direction$/i.test(line)) {
      continue
    }

    const previousValue = values.get(currentLabel)
    values.set(currentLabel, [previousValue, line].filter(Boolean).join(' '))
  }

  const getValue = (label: string) => values.get(label)?.trim() || '미정'
  const legacyBudgetAndDuration = getValue('예산/기간 범위')
  const legacyBudgetParts = splitBudgetAndDuration(legacyBudgetAndDuration)
  const projectScope = getValue('Project Scope')
  const targetTimeline = getValue('Target Timeline')
  const keyFeatures = getValue('Key Features')
  const data = {
    title: getValue('프로젝트명'),
    goal: getValue('프로젝트 목표'),
    category: getValue('제품 카테고리'),
    budgetMinimum: getValue('Budget Minimum'),
    budgetRange:
      projectScope !== '미정' ? projectScope : legacyBudgetParts.budgetRange || '미정',
    duration:
      targetTimeline !== '미정' ? targetTimeline : legacyBudgetParts.duration || '미정',
    budgetAndDuration: legacyBudgetAndDuration,
    size: getValue('예상 크기'),
    features: keyFeatures !== '미정' ? keyFeatures : getValue('주요 기능'),
    usage: getValue('최종 활용 목적'),
    ideaSummary: getValue('아이디어 정리'),
    referenceSummary: getValue('참고 자료'),
  }
  const hasMinimumData =
    data.title !== '미정' ||
    data.goal !== '미정' ||
    data.ideaSummary !== '미정'

  return hasMinimumData ? data : null
}

function stripProjectDirectionCard(content: string) {
  return content
    .replace(
      /<<AIDEE_PROJECT_DIRECTION>>\s*[\s\S]*?\s*<<\/AIDEE_PROJECT_DIRECTION>>\s*/g,
      ''
    )
    .replace(
      /#\s*Project\s*(?:Card|Direction)\s*[\s\S]*?(?=\n\s*제품의\s*구체적인\s*모습|\n\s*형태,\s*색감|\s*$)/gi,
      ''
    )
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function splitProjectDirectionContent(content: string) {
  const markerMatch = content.match(
    /<<AIDEE_PROJECT_DIRECTION>>\s*[\s\S]*?\s*<<\/AIDEE_PROJECT_DIRECTION>>/
  )
  const match =
    markerMatch ??
    content.match(
      /#\s*Project\s*(?:Card|Direction)\s*[\s\S]*?(?=\n\s*제품의\s*구체적인\s*모습|\n\s*형태,\s*색감|\s*$)/i
    )

  if (!match || typeof match.index !== 'number') {
    return {
      before: '',
      after: stripProjectDirectionCard(content),
    }
  }

  const before = content.slice(0, match.index).trim()
  const after = content.slice(match.index + match[0].length).trim()

  return {
    before,
    after,
  }
}

function getPersonaArtifactKind(content: string): PersonaArtifactKind | null {
  if (/##\s*Problem Statements/i.test(content)) {
    return 'problem_statements'
  }

  if (/##\s*Keywords:\s*Experience/i.test(content)) {
    return 'experience_keywords'
  }

  if (/##\s*Keywords:\s*Relationship/i.test(content)) {
    return 'relationship_keywords'
  }

  if (
    /##\s*Persona Summary/i.test(content) ||
    /Demographic Info|Persona Story|Problem & Needs|Current Behavior|Lifestyle Context|Relationship Keyword/i.test(
      content
    )
  ) {
    return 'persona'
  }

  return null
}

function getPersonaArtifactForce(
  kind: PersonaArtifactKind
): ForceImageGeneration {
  switch (kind) {
    case 'problem_statements':
      return 'problem_statements_visualization'
    case 'experience_keywords':
      return 'experience_keywords_visualization'
    case 'relationship_keywords':
      return 'relationship_keywords_visualization'
    case 'persona':
      return 'persona_visualization'
  }
}

function extractPersonaFlowCard(content: string):
  | {
      kind: Exclude<PersonaArtifactKind, 'persona'>
      summary: string
    }
  | null {
  const match = content.match(
    /<<AIDEE_PERSONA_FLOW_CARD:(problem_statements|experience_keywords|relationship_keywords)>>\s*([\s\S]*?)\s*<<\/AIDEE_PERSONA_FLOW_CARD>>/
  )

  if (!match) {
    return null
  }

  return {
    kind: match[1] as Exclude<PersonaArtifactKind, 'persona'>,
    summary: match[2].trim(),
  }
}

function stripPersonaFlowCard(content: string) {
  return content
    .replace(
      /<<AIDEE_PERSONA_FLOW_CARD:(?:problem_statements|experience_keywords|relationship_keywords)>>\s*[\s\S]*?\s*<<\/AIDEE_PERSONA_FLOW_CARD>>\s*/g,
      ''
    )
    .trim()
}

function hasDirectionWidgets(content: string) {
  return /<<AIDEE_DIRECTION_WIDGETS>>[\s\S]*?<<\/AIDEE_DIRECTION_WIDGETS>>/.test(
    content
  )
}

function extractDirectionCard(content: string):
  | {
      kind: DirectionArtifactKind
      summary: string
    }
  | null {
  const match = content.match(
    /<<AIDEE_DIRECTION_CARD:(market_size|consumption_keywords|brand_positioning)>>\s*([\s\S]*?)\s*<<\/AIDEE_DIRECTION_CARD>>/
  )

  if (!match) {
    return null
  }

  return {
    kind: match[1] as DirectionArtifactKind,
    summary: match[2].trim(),
  }
}

function stripDirectionInternalBlocks(content: string) {
  return content
    .replace(
      /<<AIDEE_DIRECTION_WIDGETS>>[\s\S]*?<<\/AIDEE_DIRECTION_WIDGETS>>\s*/g,
      ''
    )
    .replace(
      /<<AIDEE_DIRECTION_CARD:(?:market_size|consumption_keywords|brand_positioning)>>\s*[\s\S]*?\s*<<\/AIDEE_DIRECTION_CARD>>\s*/g,
      ''
    )
    .trim()
}

function getDirectionResearchKind(content: string): DirectionArtifactKind | null {
  if (/##\s*(?:시장\s*규모\s*리서치|Tam\s*Sam\s*Som)/i.test(content)) {
    return 'market_size'
  }

  if (/##\s*(?:소비\s*트렌드\s*리서치|Keywords\s*:\s*Consumption)/i.test(content)) {
    return 'consumption_keywords'
  }

  if (/##\s*(?:경쟁사\s*리서치|Positioning\s*Map\s*:\s*Brand)/i.test(content)) {
    return 'brand_positioning'
  }

  return null
}

function getDirectionArtifactForce(
  kind: DirectionArtifactKind
): ForceImageGeneration {
  switch (kind) {
    case 'market_size':
      return 'market_size_visualization'
    case 'consumption_keywords':
      return 'consumption_keywords_visualization'
    case 'brand_positioning':
      return 'brand_positioning_visualization'
  }
}

function hasStyleKeywordPicker(content: string) {
  return /<<AIDEE_STYLE_KEYWORD_PICKER>>[\s\S]*?<<\/AIDEE_STYLE_KEYWORD_PICKER>>/.test(
    content
  )
}

function stripStyleKeywordPicker(content: string) {
  return content
    .replace(
      /<<AIDEE_STYLE_KEYWORD_PICKER>>[\s\S]*?<<\/AIDEE_STYLE_KEYWORD_PICKER>>\s*/g,
      ''
    )
    .trim()
}

function isStyleReferenceProposal(content: string) {
  return /##\s*선택한\s*스타일\s*레퍼런스/i.test(content)
}

function isPersonaClarificationQuestionText(content: string) {
  const normalized = content.replace(/\s+/g, ' ').trim()
  const hasHintChoices = splitAssistantChoices(content).choices.length > 0

  return (
    hasHintChoices &&
    /(?:페르소나\s*카드를\s*만들기\s*전에|사용자를\s*조금\s*더\s*구체화|사용자의\s*기본\s*윤곽|사용\s*장면이\s*필요|불편함을\s*좁히|문제\s*상황까지\s*잡혔|나이대와\s*직업|가장\s*필요로\s*하는\s*순간|크게\s*불편해하는\s*점|가장\s*먼저\s*볼\s*기준)/.test(
      normalized
    )
  )
}

function isPersonaSummaryText(content: string) {
  if (isPersonaClarificationQuestionText(content)) {
    return false
  }

  return (
    Boolean(getPersonaArtifactKind(content)) ||
    /사용자\s*정리|사용\s*상황|핵심\s*문제|성공\s*기준|선택\s*기준/.test(
      content
    ) ||
    content.includes('Persona Card') ||
    (content.includes('User') &&
      content.includes('Problem') &&
      content.includes('Decision'))
  )
}

function isLikelyImageGenerationTurn({
  messages,
  stageKey,
}: {
  messages: ChatMessage[]
  stageKey: StageKey
}) {
  const lastUserMessage =
    [...messages].reverse().find((message) => message.role === 'user')?.content ??
    ''

  if (stageKey === 'step_4_style') {
    return /스타일\s*키워드\s*선택\s*완료|감정\s*키워드|색감\s*키워드|형태\s*키워드|촉감\s*키워드|시각화하기/i.test(
      lastUserMessage
    )
  }

  if (stageKey === 'step_5_design') {
    return true
  }

  return /이미지|시안|렌더|무드보드|비주얼|visual|render|image|그려|보여줘|만들어줘/i.test(
    lastUserMessage
  )
}

function parsePersonaData(content: string) {
  const normalizedContent = content.replace(/\r\n/g, '\n')

  const imageUrlMatch = normalizedContent.match(
    /https?:\/\/[^\s]+(?:\.jpg|\.jpeg|\.png|\.webp|\/seed\/[^\s]+)/i
  )
  const imageUrl = imageUrlMatch ? imageUrlMatch[0] : ''

  const sectionGroups = [
    ['A\\.\\s*User', '1\\.\\s*User', 'User'],
    ['B\\.\\s*Behavior\\s*Map', '2\\.\\s*Behavior\\s*Map', 'Behavior\\s*Map'],
    [
      'C\\.\\s*Correlation\\s*Analysis',
      '3\\.\\s*Correlation\\s*Analysis',
      'Correlation\\s*Analysis',
    ],
    ['D\\.\\s*Problem', '4\\.\\s*Problem', 'Problem'],
    ['E\\.\\s*Success', '5\\.\\s*Success', 'Success'],
    ['F\\.\\s*Decision', '5\\.\\s*Decision', 'Decision'],
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
    /^선택지\s*:?\s*$/.test(line) ||
    /^(?:A|B|C)\.\s*(?:리서치 진행|페르소나 수정|다시 정리)\s*$/.test(
      line
    ) ||
    /^(?:A|B|C)\.\s*.+선택 시\s*:?\s*$/.test(line) ||
    /^(?:A|B|C|D|E|F)\.\s*(?:User|Behavior Map|Correlation Analysis|Problem|Success|Decision)$/i.test(
      line
    ) ||
    /^(?:페르소나 확정|확정 정보만으로 리서치 수행|리서치 출력 완료 이후 다음 STEP으로 이동|STEP 2로 복귀|필요한 항목만 재질문|수정 반영 후 카드 재출력|동일 질문 1개 재수행|현재 정보만 한 번 더 요약|추가 질문 1개만 이어서 수행)$/.test(
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

  const extractSection = (titlePatterns: string[]) => {
    const nextTitles = sectionGroups
      .flat()
      .filter((item) => !titlePatterns.includes(item))
      .join('|')

    for (const titlePattern of titlePatterns) {
      const regex = new RegExp(
        `(?:^|\\n)\\s*(?:##+\\s*)?(?:\\*\\*\\s*)?${titlePattern}(?:\\s*\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:##+\\s*)?(?:\\*\\*\\s*)?(?:${nextTitles})(?:\\s*\\*\\*)?\\s*\\n|$)`,
        'i'
      )
      const match = normalizedContent.match(regex)
      if (match) {
        return normalizeLines(match[1])
      }
    }

    return []
  }

  const successLines = extractSection(['E\\.\\s*Success', '5\\.\\s*Success', 'Success'])
  const genericSuccessLabels = new Set([
    'E. Success',
    'Success',
    '핵심가치',
    '기대효과',
    '기대효과2',
  ])
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

      const [tag, ...descParts] = line.split(/→|:/).map((s) => s.trim())
      const normalizedTag = tag
        .replace(/^#/, '')
        .replace(/^[-•]\s*/, '')
        .trim()
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

  const userLines = extractSection(['A\\.\\s*User', '1\\.\\s*User', 'User'])
  const behaviorMapLines = extractSection([
    'B\\.\\s*Behavior\\s*Map',
    '2\\.\\s*Behavior\\s*Map',
    'Behavior\\s*Map',
  ])
  const correlationAnalysisLines = extractSection([
    'C\\.\\s*Correlation\\s*Analysis',
    '3\\.\\s*Correlation\\s*Analysis',
    'Correlation\\s*Analysis',
  ])
  const problemLines = extractSection(['D\\.\\s*Problem', '4\\.\\s*Problem', 'Problem'])
  const decisionLines = extractSection([
    'F\\.\\s*Decision',
    '5\\.\\s*Decision',
    'Decision',
  ])

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

function parsePersonaSummaryData(content: string) {
  const normalizedContent = content.replace(/\r\n/g, '\n')
  const finalPersonaLabels = [
    'Demographic Info',
    'Persona Story',
    'Problem & Needs',
    'Current Behavior',
    'Lifestyle Context',
    'Relationship Keyword',
  ]
  const sectionLabels = [
    '사용자 정리',
    '사용 상황',
    '핵심 문제',
    '성공 기준',
    '선택 기준',
  ]

  const cleanLine = (line: string) =>
    line
      .replace(/^#{1,6}\s*/, '')
      .replace(/\*\*/g, '')
      .replace(/^[-•]\s*/, '')
      .replace(/\s+/g, ' ')
      .trim()

  const extractLabeledSection = (label: string, labels: string[]) => {
    const escapedLabels = labels.map((sectionLabel) =>
      sectionLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    )
    const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const otherLabels = escapedLabels
      .filter((sectionLabel) => sectionLabel !== escapedLabel)
      .join('|')
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${escapedLabel}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${otherLabels})(?:\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalizedContent.match(regex)

    if (!match) {
      return []
    }

    return match[1]
      .split('\n')
      .map(cleanLine)
      .filter(Boolean)
      .filter((line) => !labels.includes(line))
      .filter((line) => !/시각화하기 버튼|Persona Card가 만들어집니다/.test(line))
  }

  if (/Demographic Info|Persona Story|Problem & Needs/i.test(normalizedContent)) {
    const demographicInfo = extractLabeledSection(
      'Demographic Info',
      finalPersonaLabels
    )
    const personaStory = extractLabeledSection('Persona Story', finalPersonaLabels)
    const problemNeeds = extractLabeledSection(
      'Problem & Needs',
      finalPersonaLabels
    )
    const currentBehavior = extractLabeledSection(
      'Current Behavior',
      finalPersonaLabels
    )
    const lifestyleContext = extractLabeledSection(
      'Lifestyle Context',
      finalPersonaLabels
    )
    const relationshipKeyword = extractLabeledSection(
      'Relationship Keyword',
      finalPersonaLabels
    )
    const hasMinimumData =
      demographicInfo.length > 0 ||
      personaStory.length > 0 ||
      problemNeeds.length > 0 ||
      currentBehavior.length > 0 ||
      lifestyleContext.length > 0 ||
      relationshipKeyword.length > 0

    if (!hasMinimumData) {
      return null
    }

    return {
      user: demographicInfo,
      behaviorMap: currentBehavior,
      correlationAnalysis: relationshipKeyword,
      problem: problemNeeds,
      success: personaStory.map((tag) => ({
        tag,
        desc: '',
      })),
      decision: lifestyleContext,
      demographicInfo,
      personaStory,
      problemNeeds,
      currentBehavior,
      lifestyleContext,
      relationshipKeyword,
      imageUrl: '',
    }
  }

  const extractSection = (label: string) => {
    const otherLabels = sectionLabels
      .filter((sectionLabel) => sectionLabel !== label)
      .join('|')
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${label}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${otherLabels})(?:\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalizedContent.match(regex)

    if (!match) {
      return []
    }

    return match[1]
      .split('\n')
      .map(cleanLine)
      .filter(Boolean)
      .filter((line) => !sectionLabels.includes(line))
  }

  const user = extractSection('사용자 정리')
  const behaviorMap = extractSection('사용 상황')
  const problem = extractSection('핵심 문제')
  const success = extractSection('성공 기준').map((tag) => ({
    tag,
    desc: '',
  }))
  const decision = extractSection('선택 기준')

  const hasMinimumData =
    user.length > 0 ||
    behaviorMap.length > 0 ||
    problem.length > 0 ||
    success.length > 0 ||
    decision.length > 0

  if (!hasMinimumData) {
    return null
  }

  return {
    user,
    behaviorMap,
    correlationAnalysis: ['사용 맥락 기반 니즈'],
    problem,
    success,
    decision,
    imageUrl: '',
  }
}

function parsePersonaVisualData(content: string) {
  return parsePersonaData(content) ?? parsePersonaSummaryData(content)
}

function limitProjectDirectionWords(text: string, maxWords: number) {
  const words = text.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean)

  if (words.length <= maxWords) {
    return text.replace(/\s+/g, ' ').trim()
  }

  return `${words.slice(0, maxWords).join(' ')}...`
}

function formatProjectDirectionSentence(text: string) {
  const summary = limitProjectDirectionWords(text, 100)

  if (!summary) {
    return '아직 아이디어 텍스트가 충분히 입력되지 않았습니다.'
  }

  if (/[.!?。！？]$/.test(summary)) {
    return summary
  }

  if (/(다|요|니다|습니다)$/.test(summary)) {
    return `${summary}.`
  }

  return `${summary}가 목표입니다.`
}

function formatProjectScope(value: string) {
  return (
    value
      .replace(/\s*~\s*/g, ' - ')
      .replace(/\s*-\s*/g, ' - ')
      .replace(/만\s*원/g, '만원')
      .replace(/억\s*원/g, '억원')
      .trim() || '미정'
  )
}

function parseProjectScopeMinimum(value: string) {
  const firstValue = value.split(/\s*(?:-|~|\/)\s*/)[0] ?? ''
  const eokMatch = firstValue.match(/([\d,.]+)\s*억/)
  const manwonMatch = firstValue.match(/([\d,.]+)\s*만\s*원?/)
  const plainMatch = firstValue.match(/([\d,.]+)/)

  if (eokMatch) {
    return Number(eokMatch[1].replace(/,/g, '')) * 10000
  }

  if (manwonMatch) {
    return Number(manwonMatch[1].replace(/,/g, ''))
  }

  if (plainMatch) {
    return Number(plainMatch[1].replace(/,/g, ''))
  }

  return Number.NaN
}

function formatProjectMinimumBudget(data: ProjectDirectionData) {
  if (data.budgetMinimum && data.budgetMinimum !== '미정') {
    return data.budgetMinimum
  }

  const minBudget = parseProjectScopeMinimum(data.budgetRange)

  if (Number.isNaN(minBudget)) {
    return '미정'
  }

  return `$${Math.round(minBudget / 100)}K+`
}

function formatProjectTimeline(value: string) {
  const normalized = value.replace(/\s+/g, '')
  const yearMatch = normalized.match(/(\d+(?:\.\d+)?)년/)
  const monthMatch = normalized.match(/(\d+(?:\.\d+)?)(?:개월|달|months?|mo)/i)
  const weekMatch = normalized.match(/(\d+(?:\.\d+)?)(?:주|weeks?|w)/i)

  if (/months?|weeks?|years?/i.test(value)) {
    return value
  }

  if (yearMatch) {
    return `${Number(yearMatch[1]) * 12}${normalized.includes('+') ? '+' : ''} Months`
  }

  if (monthMatch) {
    const months = Number(monthMatch[1])

    return `${months}${normalized.includes('+') ? '+' : ''} ${
      months === 1 ? 'Month' : 'Months'
    }`
  }

  if (weekMatch) {
    const weeks = Number(weekMatch[1])

    return `${weeks}${normalized.includes('+') ? '+' : ''} ${
      weeks === 1 ? 'Week' : 'Weeks'
    }`
  }

  return value || '미정'
}

function getProjectFeatureItems(value: string) {
  const items = value
    .split(/\n|,|，|、/g)
    .map((item) => item.replace(/^[-•]\s*/, '').trim())
    .filter(Boolean)

  return items.length > 0 ? items : ['미정']
}

function ProjectDirectionCard({ data }: { data: ProjectDirectionData }) {
  const featureItems = getProjectFeatureItems(data.features)
  const projectScope = formatProjectScope(data.budgetRange)
  const minimumBudget = formatProjectMinimumBudget(data)
  const targetTimeline = formatProjectTimeline(data.duration)
  const ideaSummary = formatProjectDirectionSentence(data.ideaSummary)

  return (
    <div className="my-4 w-full max-w-[684px] overflow-x-auto pb-1">
      <div className="relative h-[332px] w-[560px] overflow-hidden rounded-xl bg-white font-sans shadow-[0px_0px_24px_0px_rgba(0,0,0,0.12)]">
        <div className="absolute left-0 top-0 flex h-full w-36 flex-col justify-between bg-zinc-200 px-5 py-5">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-500">
              Start Point
            </p>
            <h2 className="mt-2 text-xl font-bold leading-6 text-zinc-700">
              Project Direction
            </h2>
          </div>
          <div>
            <p className="text-[10px] font-semibold text-zinc-500">프로젝트명</p>
            <p className="mt-1 max-h-12 overflow-hidden text-sm font-bold leading-5 text-zinc-800">
              {data.title}
            </p>
          </div>
        </div>

        <div className="absolute left-[166px] top-[18px] h-[296px] w-[370px] overflow-hidden">
          <p className="max-h-[24px] overflow-hidden text-[15px] font-bold leading-6 text-zinc-800">
            {data.category}
          </p>
          <p className="mt-2 max-h-[54px] overflow-hidden text-[12px] font-medium leading-[18px] text-zinc-600">
            {ideaSummary}
          </p>

          <div className="mt-4 grid grid-cols-[132px_1fr] gap-x-4">
            <div>
              <p className="max-h-[36px] overflow-hidden text-[30px] font-extrabold leading-9 text-zinc-900">
                {minimumBudget}
              </p>

              <div className="mt-5">
                <ProjectDirectionSection
                  label="Target Timeline"
                  value={targetTimeline}
                />
              </div>
            </div>

            <div className="min-w-0">
              <ProjectDirectionSection
                label="Project Scope"
                value={projectScope}
              />

              <div className="mt-4">
                <p className="text-[9px] font-semibold uppercase leading-3 tracking-[0.1em] text-zinc-400">
                  Key Features
                </p>
                <ul className="mt-1.5 max-h-[94px] space-y-1 overflow-hidden text-[11px] font-medium leading-[15px] text-zinc-700">
                  {featureItems.map((feature) => (
                    <li key={feature} className="flex gap-1.5">
                      <span className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-zinc-400" />
                      <span className="min-w-0 break-words">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ProjectDirectionSection({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div>
      <p className="text-[9px] font-semibold uppercase leading-3 tracking-[0.1em] text-zinc-400">
        {label}
      </p>
      <p className="mt-1 max-h-[34px] overflow-hidden break-words text-[12px] font-bold leading-[17px] text-zinc-700">
        {value}
      </p>
    </div>
  )
}

type ProblemStatementCardSection = {
  label: '01. Context' | '02. Problems' | '03. Needs'
  title: string
  description: string
}

function getPersonaFlowBody(
  summary: string,
  title: string
) {
  return summary
    .replace(new RegExp(`^##\\s*${title}\\s*`, 'i'), '')
    .split('\n')
    .filter((line) => !/아래의 시각화하기 버튼/.test(line))
    .join('\n')
    .trim()
}

function cleanProblemStatementLine(line: string) {
  return line
    .replace(/^#{1,6}\s*/, '')
    .replace(/^[-•]\s*/, '')
    .replace(/\*\*/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function getProblemStatementLabel(line: string):
  | ProblemStatementCardSection['label']
  | null {
  const normalized = line.toLowerCase()

  if (/^(?:0?1[.)]\s*)?context$/.test(normalized) || line === '문제(현재 상황)') {
    return '01. Context'
  }

  if (/^(?:0?2[.)]\s*)?problems?$/.test(normalized) || line === '불편함') {
    return '02. Problems'
  }

  if (/^(?:0?3[.)]\s*)?needs?$/.test(normalized)) {
    return '03. Needs'
  }

  return null
}

function normalizeProblemStatementText(text: string) {
  return text.replace(/[\s.,:;!?。！？'"]/g, '').trim()
}

function cleanupProblemStatementTitle(text: string) {
  return text
    .replace(/\.\.\.$/, '')
    .replace(/^(사용자는|사용자가|기존 방식은|실제)\s*/, '')
    .replace(/(?:상황을\s*다룬다|문제가\s*남는다|필요하다|원한다|기반으로\s*한다|이어진다)$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function isGenericProblemStatementTitle(title: string) {
  return /^(?:사용 맥락|문제 흐름|필요 방향|Context|Problems|Needs)(?:\s*요약)?$|^수정된\s*(?:사용 맥락|문제 흐름|필요 방향)$/i.test(
    title.trim()
  )
}

function summarizeProblemStatementTitle(description: string, fallback: string) {
  const firstSentence = description
    .replace(/\s+/g, ' ')
    .trim()
    .split(/[.!?。！？]/)[0]
    ?.trim()
  const colonTitle = firstSentence?.split(/[:：]/)[0]?.trim() || ''
  const title = cleanupProblemStatementTitle(colonTitle)

  if (
    !title ||
    normalizeProblemStatementText(title) === normalizeProblemStatementText(description)
  ) {
    return fallback
  }

  return title
}

function stripProblemStatementTitlePrefix(description: string, title: string) {
  return description
    .replace(new RegExp(`^\\s*${escapeRegExp(title)}\\s*[:：]\\s*`, 'i'), '')
    .trim()
}

function isDuplicateProblemStatementTitle(title: string, description: string) {
  const normalizedTitle = normalizeProblemStatementText(title)
  const normalizedDescription = normalizeProblemStatementText(description)

  return (
    normalizedTitle.length > 0 &&
    (normalizedTitle === normalizedDescription ||
      normalizedDescription.startsWith(normalizedTitle))
  )
}

function parseProblemStatementSections(summary: string) {
  const body = getPersonaFlowBody(summary, 'Problem Statements')
  const sections: ProblemStatementCardSection[] = []
  let current: ProblemStatementCardSection | null = null

  const pushCurrent = () => {
    if (!current) {
      return
    }
    const fallbackTitle = current.label.replace(/^\d+\.\s*/, '')
    const title = current.title.trim()
    const rawDescription =
      current.description.trim() || title || '내용을 정리 중입니다.'
    const shouldSummarizeTitle =
      !title ||
      !current.description.trim() ||
      isGenericProblemStatementTitle(title) ||
      isDuplicateProblemStatementTitle(title, rawDescription)
    const sectionTitle = shouldSummarizeTitle
      ? summarizeProblemStatementTitle(rawDescription, fallbackTitle)
      : cleanupProblemStatementTitle(title)
    const description =
      stripProblemStatementTitlePrefix(rawDescription, sectionTitle) ||
      rawDescription

    sections.push({
      label: current.label,
      title: sectionTitle,
      description,
    })
  }

  body
    .split('\n')
    .map(cleanProblemStatementLine)
    .filter(Boolean)
    .forEach((line) => {
      const label = getProblemStatementLabel(line)

      if (label) {
        pushCurrent()
        current = { label, title: '', description: '' }
        return
      }

      if (!current) {
        return
      }

      if (!current.title) {
        current.title = line
        return
      }

      current.description = [current.description, line].filter(Boolean).join(' ')
    })

  pushCurrent()

  const defaults: ProblemStatementCardSection[] = [
    {
      label: '01. Context',
      title: '사용 맥락',
      description: '사용자가 제품을 필요로 하는 장면과 생활 패턴을 정리합니다.',
    },
    {
      label: '02. Problems',
      title: '문제 흐름',
      description: '현재 방식에서 반복되는 불편과 해결되지 않는 문제를 정리합니다.',
    },
    {
      label: '03. Needs',
      title: '필요 방향',
      description: '사용자가 기대하는 변화와 제품이 제공해야 할 지원을 정리합니다.',
    },
  ]

  return defaults.map(
    (fallback) =>
      sections.find((section) => section.label === fallback.label) ?? fallback
  )
}

function ProblemStatementsFlowCard({ summary }: { summary: string }) {
  const sections = parseProblemStatementSections(summary)

  return (
    <div className="my-3 w-full max-w-[602px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-neutral-900">
          Problem Statements
        </h3>
        <span className="rounded-full border border-blue-200 bg-blue-50/60 px-2.5 py-1 text-[11px] font-bold text-blue-700">
          STEP 2
        </span>
      </div>
      <div className="space-y-3">
        {sections.map((section) => (
          <section
            key={section.label}
            className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0"
          >
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-blue-600">
              {section.label}
            </p>
            <h4 className="mt-1 text-sm font-bold leading-5 text-neutral-900">
              {section.title}
            </h4>
            <p className="mt-1 text-sm leading-6 text-slate-700">
              {section.description}
            </p>
          </section>
        ))}
      </div>
    </div>
  )
}

type ExperienceKeywordsCardData = {
  intro: string
  description: string
  keywords: string[]
}

function parseCommaKeywords(text: string) {
  return text
    .split(/[,，、/]|(?:\s*·\s*)/g)
    .map((item) =>
      item
        .replace(/^[-•]\s*/, '')
        .replace(/\*\*/g, '')
        .replace(/\s+/g, ' ')
        .trim()
    )
    .filter(Boolean)
}

function uniqueKeywordItems(items: string[]) {
  const seen = new Set<string>()
  const keywords: string[] = []

  for (const item of items) {
    const normalized = item.replace(/\s+/g, '').toLowerCase()

    if (!normalized || seen.has(normalized)) {
      continue
    }

    seen.add(normalized)
    keywords.push(item)
  }

  return keywords
}

function buildExperienceIntroSentence(keywords: string[]) {
  const [
    context = '직관적인 사용 흐름',
    valueA = '신뢰감',
    valueB = '유연함',
    outcome = '지속 가능한 변화',
  ] = keywords

  return `${context} 속에서, ${valueA}와 ${valueB}으로 ${outcome}을 경험하는 제품`
}

function extractKeywordSection(
  summary: string,
  labels: string[],
  boundaryLabels: string[]
) {
  const normalized = summary.replace(/\r\n/g, '\n')
  const escape = (value: string) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const boundary = boundaryLabels.map(escape).join('|')

  for (const label of labels) {
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${escape(
        label
      )}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${boundary})(?:\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalized.match(regex)

    if (match?.[1]) {
      return parseCommaKeywords(match[1])
    }
  }

  return []
}

function extractSingleTextSection(
  summary: string,
  labels: string[],
  boundaryLabels: string[]
) {
  const normalized = summary.replace(/\r\n/g, '\n')
  const escape = (value: string) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const boundary = boundaryLabels.map(escape).join('|')

  for (const label of labels) {
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${escape(
        label
      )}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${boundary})(?:\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalized.match(regex)

    if (match?.[1]) {
      return match[1]
        .split('\n')
        .map((line) =>
          line
            .replace(/^[-•]\s*/, '')
            .replace(/\*\*/g, '')
            .replace(/\s+/g, ' ')
            .trim()
        )
        .filter(Boolean)
        .join(' ')
    }
  }

  return ''
}

function parseExperienceKeywordsCard(summary: string): ExperienceKeywordsCardData {
  const body = getPersonaFlowBody(summary, 'Keywords: Experience')
  const sectionLabels = [
    '한줄 소개',
    '내용',
    'Keywords',
    '감정 Keywords (12)',
    '행동 Keywords (12)',
    '공간 Keywords (8)',
    '감정',
    '행동',
    '공간',
  ]
  const emotionKeywords = extractKeywordSection(
    body,
    ['감정 Keywords (12)', '감정'],
    sectionLabels
  )
  const behaviorKeywords = extractKeywordSection(
    body,
    ['행동 Keywords (12)', '행동'],
    sectionLabels
  )
  const spaceKeywords = extractKeywordSection(
    body,
    ['공간 Keywords (8)', '공간'],
    sectionLabels
  )
  const mergedKeywords = extractKeywordSection(
    body,
    ['Keywords'],
    sectionLabels
  )
  const keywords = uniqueKeywordItems([
    ...mergedKeywords,
    ...emotionKeywords,
    ...behaviorKeywords,
    ...spaceKeywords,
  ])
  const primaryKeywords = keywords.slice(0, 6)
  const introSection = extractSingleTextSection(
    body,
    ['한줄 소개'],
    sectionLabels
  )
  const descriptionSection = extractSingleTextSection(
    body,
    ['내용'],
    sectionLabels
  )
  const intro =
    introSection ||
    buildExperienceIntroSentence(primaryKeywords)
  const description =
    descriptionSection ||
    (keywords.length > 0
      ? `이 제품은 사용자가 자연스럽게 시작하고 흐름을 회복하며, 사용 후에는 안정감과 만족감을 느끼는 경험을 제공합니다. 공간 안에서는 부담 없이 놓이고 생활 흐름과 연결되는 방향을 지향하며, ${primaryKeywords.join(
          ', '
        )} 같은 감각을 중심으로 제품 경험을 제안합니다.`
      : '이 제품은 사용자가 쉽게 시작하고 자연스럽게 몰입하며, 사용 후에는 긍정적인 감정과 안정된 리듬을 느끼는 경험을 지향합니다. 제품의 기능과 공간 속 존재감이 함께 연결되는 방향을 제안합니다.')

  return {
    intro,
    description,
    keywords,
  }
}

function ExperienceKeywordsFlowCard({ summary }: { summary: string }) {
  const data = parseExperienceKeywordsCard(summary)

  return (
    <div className="my-3 w-full max-w-[602px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-neutral-900">
          Keywords: Experience
        </h3>
        <span className="rounded-full border border-emerald-200 bg-emerald-50/60 px-2.5 py-1 text-[11px] font-bold text-emerald-700">
          STEP 2
        </span>
      </div>
      <p className="text-sm font-bold leading-5 text-neutral-900">
        {data.intro}
      </p>
      <p className="mt-2 text-sm leading-6 text-slate-700">
        {data.description}
      </p>
      {data.keywords.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-1.5">
          {data.keywords.map((keyword) => (
            <span
              key={keyword}
              className="rounded-full border border-emerald-100 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700"
            >
              {keyword}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function PersonaFlowCard({
  kind,
  summary,
}: {
  kind: Exclude<PersonaArtifactKind, 'persona'>
  summary: string
}) {
  const titleMap: Record<Exclude<PersonaArtifactKind, 'persona'>, string> = {
    problem_statements: 'Problem Statements',
    experience_keywords: 'Keywords: Experience',
    relationship_keywords: 'Keywords: Relationship',
  }
  const accentClassMap: Record<
    Exclude<PersonaArtifactKind, 'persona'>,
    string
  > = {
    problem_statements: 'border-blue-200 bg-blue-50/60 text-blue-700',
    experience_keywords: 'border-emerald-200 bg-emerald-50/60 text-emerald-700',
    relationship_keywords: 'border-violet-200 bg-violet-50/60 text-violet-700',
  }

  if (kind === 'problem_statements') {
    return <ProblemStatementsFlowCard summary={summary} />
  }

  if (kind === 'experience_keywords') {
    return <ExperienceKeywordsFlowCard summary={summary} />
  }

  const body = getPersonaFlowBody(summary, titleMap[kind])

  return (
    <div className="my-3 w-full max-w-[602px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-neutral-900">
          {titleMap[kind]}
        </h3>
        <span
          className={`rounded-full border px-2.5 py-1 text-[11px] font-bold ${accentClassMap[kind]}`}
        >
          STEP 2
        </span>
      </div>
      <div className="prose prose-sm prose-p:my-0 prose-li:my-1 prose-strong:text-neutral-900 max-w-none text-sm leading-6 text-slate-700">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
          {body}
        </ReactMarkdown>
      </div>
    </div>
  )
}

function DirectionResearchWidgets({
  disabled,
  onSelect,
}: {
  disabled: boolean
  onSelect: (kind: DirectionArtifactKind) => void
}) {
  const widgets: Array<{
    kind: DirectionArtifactKind
    index: string
    title: string
    description: string
  }> = [
    {
      kind: 'market_size',
      index: '1',
      title: '시장 규모',
      description: 'TAM/SAM/SOM 관점으로 진입 시장을 봅니다.',
    },
    {
      kind: 'consumption_keywords',
      index: '2',
      title: '소비 트렌드',
      description: '구매 동기와 소비 키워드를 정리합니다.',
    },
    {
      kind: 'brand_positioning',
      index: '3',
      title: '경쟁사',
      description: '경쟁 구도와 브랜드 포지션을 비교합니다.',
    },
  ]

  return (
    <div className="my-3 grid w-full max-w-[680px] grid-cols-1 gap-3 sm:grid-cols-3">
      {widgets.map((widget) => (
        <button
          key={widget.kind}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(widget.kind)}
          className="min-h-[112px] rounded-lg border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-blue-200 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="mb-3 flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
            {widget.index}
          </span>
          <span className="block text-sm font-bold text-neutral-900">
            {widget.title}
          </span>
          <span className="mt-1 block text-xs font-medium leading-5 text-slate-500">
            {widget.description}
          </span>
        </button>
      ))}
    </div>
  )
}

function DirectionResearchCard({
  kind,
  summary,
}: {
  kind: DirectionArtifactKind
  summary: string
}) {
  const titleMap: Record<DirectionArtifactKind, string> = {
    market_size: 'Tam Sam Som',
    consumption_keywords: 'Keywords:Consumption',
    brand_positioning: 'Positioning Map: Brand',
  }
  const accentClassMap: Record<DirectionArtifactKind, string> = {
    market_size: 'border-cyan-200 bg-cyan-50/70 text-cyan-700',
    consumption_keywords:
      'border-amber-200 bg-amber-50/70 text-amber-700',
    brand_positioning: 'border-rose-200 bg-rose-50/70 text-rose-700',
  }
  const body = summary
    .replace(new RegExp(`^##\\s*${titleMap[kind]}\\s*`, 'i'), '')
    .replace(/^##\s*(?:시장\s*규모|소비\s*트렌드|경쟁사)\s*리서치\s*/i, '')
    .split('\n')
    .filter((line) => !/아래의 시각화하기 버튼/.test(line))
    .join('\n')
    .trim()

  if (kind === 'brand_positioning') {
    const getQuadrantBrands = (label: string, fallback: string[]) => {
      const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const match = body.match(
        new RegExp(
          `\\*\\*${escapedLabel}\\*\\*[\\s\\S]*?경쟁사\\s*브랜드\\s*:\\s*([^\\n]+)`,
          'i'
        )
      )

      if (!match) {
        return fallback
      }

      const parsed = match[1]
        .split(',')
        .map((brand) => brand.trim())
        .filter(Boolean)

      return parsed.length >= 2 ? parsed.slice(0, 2) : fallback
    }

    const quadrantBrands: Array<{
      title: string
      description: string
      brands: string[]
      className: string
    }> = [
      {
        title: '1사분면 · 프리미엄 라이프스타일형',
        description: '디자인 완성도 · 브랜드 감성 · 소장 가치',
        brands: getQuadrantBrands('프리미엄 라이프스타일형', [
          'Dyson',
          'Apple',
        ]),
        className: 'col-start-1 row-start-1 border-rose-200 bg-rose-50',
      },
      {
        title: '2사분면 · 합리적 라이프스타일형',
        description: '공간 분위기 · 취향 반영 · 접근 가능한 가격',
        brands: getQuadrantBrands('합리적 라이프스타일형', ['MUJI', 'IKEA']),
        className: 'col-start-2 row-start-1 border-amber-200 bg-amber-50',
      },
      {
        title: '3사분면 · 합리적 기능형',
        description: '낮은 가격 · 기본 기능 · 실용 구매',
        brands: getQuadrantBrands('합리적 기능형', ['Xiaomi', 'Govee']),
        className: 'col-start-2 row-start-2 border-sky-200 bg-sky-50',
      },
      {
        title: '4사분면 · 프리미엄 기능형',
        description: '성능 · 정확성 · 기술 신뢰도',
        brands: getQuadrantBrands('프리미엄 기능형', ['Philips Hue', 'Garmin']),
        className: 'col-start-1 row-start-2 border-violet-200 bg-violet-50',
      },
    ]

    return (
      <div className="my-3 w-full max-w-[680px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-base font-bold text-neutral-900">
            Positioning Map: Brand
          </h3>
          <span className="rounded-full border border-rose-200 bg-rose-50/70 px-2.5 py-1 text-[11px] font-bold text-rose-700">
            STEP 3
          </span>
        </div>

        <div className="mb-3 rounded-md bg-slate-50 px-3 py-2 text-xs font-semibold leading-5 text-slate-600">
          X축: 프리미엄 가격 ↔ 합리적 가격
          <br />
          Y축: 기능 중심 ↔ 라이프스타일 중심
        </div>

        <div className="relative grid min-h-[420px] grid-cols-2 grid-rows-2 gap-2 rounded-lg border border-slate-200 bg-white p-6">
          <div className="pointer-events-none absolute left-1/2 top-6 bottom-6 w-px bg-slate-300" />
          <div className="pointer-events-none absolute top-1/2 left-6 right-6 h-px bg-slate-300" />
          <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 text-[10px] font-bold text-slate-400">
            라이프스타일 중심
          </div>
          <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] font-bold text-slate-400">
            기능 중심
          </div>
          <div className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] font-bold text-slate-400">
            프리미엄 가격
          </div>
          <div className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rotate-90 text-[10px] font-bold text-slate-400">
            합리적 가격
          </div>
          {quadrantBrands.map((quadrant) => (
            <div
              key={quadrant.title}
              className={`relative rounded-md border p-3 ${quadrant.className}`}
            >
              <div className="text-sm font-bold text-neutral-900">
                {quadrant.title}
              </div>
              <div className="mt-1 text-[11px] font-medium leading-4 text-slate-500">
                {quadrant.description}
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {quadrant.brands.map((brand) => (
                  <span
                    key={brand}
                    className="rounded-full bg-white px-2.5 py-1 text-xs font-bold text-slate-700 shadow-sm"
                  >
                    {brand}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <div className="absolute left-[58%] top-[34%] rounded-full bg-blue-600 px-3 py-1.5 text-xs font-bold text-white shadow-lg">
            OUR BRAND
          </div>
        </div>

        <div className="mt-3 text-sm leading-6 text-slate-700">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
            {body}
          </ReactMarkdown>
        </div>
      </div>
    )
  }

  return (
    <div className="my-3 w-full max-w-[680px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-neutral-900">
          {titleMap[kind]}
        </h3>
        <span
          className={`rounded-full border px-2.5 py-1 text-[11px] font-bold ${accentClassMap[kind]}`}
        >
          STEP 3
        </span>
      </div>
      <div className="prose prose-sm prose-p:my-0 prose-li:my-1 prose-strong:text-neutral-900 max-w-none text-sm leading-6 text-slate-700">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
          {body}
        </ReactMarkdown>
      </div>
    </div>
  )
}

const STYLE_KEYWORD_GROUPS = [
  {
    key: 'emotion',
    title: '1. 이 제품을 사용할 때 어떤 기분이 들었으면 좋겠나요?',
    label: '감정 키워드',
    keywords: [
      '차분한',
      '편안한',
      '안정적인',
      '따뜻한',
      '고요한',
      '산뜻한',
      '경쾌한',
      '정돈된',
      '활기찬',
      '몰입감 있는',
      '섬세한',
      '부드러운',
      '세련된',
      '신뢰감 있는',
      '친근한',
      '감각적인',
      '위로가 되는',
      '자기주도적인',
      '특별한',
      '영감을 주는',
      '평온한',
      '상쾌한',
      '균형감 있는',
      '명료한',
      '절제된',
      '부담 없는',
      '집중되는',
      '개운한',
      '믿음직한',
      '자연스러운',
    ],
  },
  {
    key: 'color',
    title: '2. 제품의 색감은 어떤 분위기에 가까우면 좋을까요?',
    label: '색감 키워드',
    keywords: [
      '뉴트럴/베이지',
      '소프트 화이트',
      '모노톤',
      '비비드/원색',
      '딥/다크 톤',
      '자연의 색',
      '네온/형광',
      '따뜻한',
      '톤온톤',
      '투명한',
      '뮤트 톤',
      '메탈릭',
      '그라데이션',
      '웜톤',
      '쿨톤',
      '포인트 컬러',
      '빈티지/세피아',
      '세이지 그린',
      '더스티 블루',
      '붉은색',
      '웜 그레이',
      '쿨 그레이',
      '오프화이트',
      '크림 톤',
      '차콜',
      '실버',
      '코랄',
      '올리브',
      '파우더 핑크',
      '우드 톤',
    ],
  },
  {
    key: 'shape',
    title: '3. 제품의 형태는 어떤 인상에 가까우면 좋을까요?',
    label: '형태 키워드',
    keywords: [
      '미니멀한',
      '부드러운 곡선',
      '정제된',
      '단순한',
      '둥근',
      '절제된 직선',
      '안정적인 비례',
      '대칭적인',
      '컴팩트한',
      '비대칭적인',
      '슬림한',
      '정밀한',
      '일체형',
      '맥시멀한',
      '유기적인',
      '기하학적인',
      '오브제형',
      '손에 잡히는',
      '조형적인',
      '간결한',
      '유선형',
      '모듈형',
      '아이코닉',
      '직관적인',
      '공간 절약',
      '스탠드형',
      '플랫한',
      '입체적인',
      '균형 잡힌',
      '가벼워 보이는',
    ],
  },
  {
    key: 'touch',
    title: '4. 제품의 표면과 촉감은 어떤 느낌이면 좋을까요?',
    label: '촉감 키워드',
    keywords: [
      '매트한',
      '소프트 터치',
      '실키한',
      '단단한',
      '가벼운',
      '따뜻한',
      '차가운',
      '반투명한',
      '투명한',
      '빛을 확산하는',
      '은은하게 빛나는',
      '세라믹 같은',
      '알루미늄 느낌',
      '미세한 텍스처',
      '패브릭 느낌',
      '우드 느낌',
      '고무 코팅 느낌',
      '메탈릭한',
      '매끄러운',
      '내구성 있는',
      '보송한',
      '무광 플라스틱',
      '미끄럼 방지',
      '그립감 있는',
      '탄성 있는',
      '묵직한',
      '자연 질감',
      '생활 방수',
      '스크래치에 강한',
      '지문이 덜 남는',
    ],
  },
] as const

function StyleKeywordPicker({
  disabled,
  onSubmit,
}: {
  disabled: boolean
  onSubmit: (text: string) => void
}) {
  const [selectedKeywords, setSelectedKeywords] = useState<
    Record<string, string[]>
  >({})
  const maxSelection = 5

  const toggleKeyword = (groupKey: string, keyword: string) => {
    setSelectedKeywords((prev) => {
      const current = prev[groupKey] ?? []
      const exists = current.includes(keyword)
      const next = exists
        ? current.filter((item) => item !== keyword)
        : current.length >= maxSelection
          ? current
          : [...current, keyword]

      return {
        ...prev,
        [groupKey]: next,
      }
    })
  }

  const canSubmit = STYLE_KEYWORD_GROUPS.every(
    (group) => (selectedKeywords[group.key] ?? []).length > 0
  )

  const submit = () => {
    if (!canSubmit || disabled) {
      return
    }

    const summary = STYLE_KEYWORD_GROUPS.map((group) => {
      const values = selectedKeywords[group.key] ?? []
      return `${group.label}: ${values.join(', ')}`
    }).join('\n')

    onSubmit(`스타일 키워드 선택 완료\n${summary}`)
  }

  return (
    <div className="my-3 w-full max-w-[760px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="space-y-5">
        {STYLE_KEYWORD_GROUPS.map((group) => {
          const selected = selectedKeywords[group.key] ?? []

          return (
            <section key={group.key} className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-bold text-neutral-900">
                  {group.title}
                </h3>
                <span className="text-xs font-semibold text-slate-400">
                  {selected.length}/{maxSelection}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {group.keywords.map((keyword) => {
                  const isSelected = selected.includes(keyword)
                  const isBlocked =
                    !isSelected && selected.length >= maxSelection

                  return (
                    <button
                      key={`${group.key}-${keyword}`}
                      type="button"
                      disabled={disabled || isBlocked}
                      onClick={() => toggleKeyword(group.key, keyword)}
                      className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                        isSelected
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-100 text-slate-600 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-40'
                      }`}
                    >
                      {keyword}
                    </button>
                  )
                })}
              </div>
            </section>
          )
        })}
      </div>
      <div className="mt-4 flex justify-end">
        <button
          type="button"
          disabled={disabled || !canSubmit}
          onClick={submit}
          className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          스타일 분위기 제안받기
        </button>
      </div>
    </div>
  )
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
    case 'step_0_start':
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
    '개발 조건 정리'
  )
}

function getStageSignature(stageKey: StageKey) {
  return `${stageKey}:${getStageExperts(stageKey).join(',')}`
}

function StageDivider({ stageKey }: { stageKey: StageKey }) {
  if (stageKey === 'step_0_start') {
    return null
  }

  const experts = getStageExperts(stageKey)
  const expertLabels = experts
    .map((expert) => getExpertDefinition(expert).label)
    .join(', ')

  return (
    <div className="flex items-center gap-2 py-1" data-stage-divider={stageKey}>
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

function SidebarExpertRow({
  expertKey,
  selected,
}: {
  expertKey: ExpertKey
  selected: boolean
}) {
  const definition = getExpertDefinition(expertKey)

  return (
    <div
      className={`inline-flex w-full items-center gap-2 rounded-xl p-1.5 transition ${
        selected ? 'bg-blue-50' : 'hover:bg-gray-50'
      }`}
    >
      <ExpertAvatar
        expertKey={expertKey}
        selected={selected}
        className="h-8 w-8"
      />
      <div
        className={`text-sm font-medium leading-5 ${
          selected ? 'text-neutral-900' : 'text-neutral-600'
        }`}
      >
        {definition.label}
      </div>
    </div>
  )
}

function buildCompanyRecommendations(
  rfpJson: RfpDocument | null,
  rfpContent: string | null
): CompanyRecommendation[] {
  const styleHint =
    rfpJson?.styleKeywords.slice(0, 2).join(' · ') ||
    (rfpContent ? '기획안 기준' : '프로젝트 기준')
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
      thumbnail: '/images/partner-company-1.png',
    },
    {
      name: '상상제작소',
      summary: `${styleHint} 시제품 제작 / 목업 대응`,
      website: 'example.com',
      highlight: '핵심: 기구·회로·제작',
      thumbnail: '/images/partner-company-2.png',
    },
    {
      name: '(주) 한국기술',
      summary: `${styleHint} 양산성 검토 / 제작 파트너`,
      website: 'example.com',
      highlight: '핵심: 제작 안정성',
      thumbnail: '/images/partner-company-3.png',
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
        마지막은 시제품 제작에 필요한 협력 파트너를 추천해드리는 시간입니다.
        앞서 정리한 요구사항과 디자인 방향을 기준으로 3곳을 제안합니다.
      </div>

      <div className="flex flex-wrap gap-5">
        {companies.map((company) => (
          <div
            key={company.name}
            className="w-[240px] overflow-hidden rounded-2xl border border-gray-300 bg-white shadow-sm"
          >
            <div className="h-36 overflow-hidden bg-slate-100">
              <img
                src={company.thumbnail}
                alt={`${company.name} thumbnail`}
                className="h-full w-full object-cover"
              />
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
        파트너 더 보기
        <span className="text-blue-500">›</span>
      </button>
    </div>
  )
}

function RfpActionPanel({
  isDownloadingRfp,
  isLoading,
  onDownload,
}: {
  isDownloadingRfp: boolean
  isLoading: boolean
  onDownload: () => void
}) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onDownload}
        disabled={isDownloadingRfp || isLoading}
        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isDownloadingRfp ? '기획안 생성 중...' : '기획안 다운로드'}
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
  const [loadingLabelOverride, setLoadingLabelOverride] = useState<string | null>(
    null
  )
  const [isInitialLoading, setIsInitialLoading] = useState(true)
  const [isDownloadingRfp, setIsDownloadingRfp] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [currentStageKey, setCurrentStageKey] = useState<StageKey>('step_0_start')
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
  const [pendingNextStageKey, setPendingNextStageKey] =
    useState<StageKey | null>(null)
  const [hintModalMessageId, setHintModalMessageId] = useState<string | null>(
    null
  )
  const [confirmedPersonaMessageIds, setConfirmedPersonaMessageIds] = useState<
    Record<string, boolean>
  >({})
  const [confirmedPersonaFlowCardIds, setConfirmedPersonaFlowCardIds] = useState<
    Record<string, boolean>
  >({})
  const [confirmedVisualizationMessageIds, setConfirmedVisualizationMessageIds] =
    useState<Record<string, boolean>>({})
  const [visualizedPersonaMessageIds, setVisualizedPersonaMessageIds] = useState<
    Record<string, boolean>
  >({})
  const [visualizedDirectionMessageIds, setVisualizedDirectionMessageIds] =
    useState<Record<string, boolean>>({})
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const searchParams = useSearchParams()
  const activeExpertDefinition = getExpertDefinition(activeExpert)
  const selectableExperts = EXPERT_DEFINITIONS.filter(
    (expert) => expert.key !== 'aidee'
  )
  const uiStageKey = pendingNextStageKey ?? currentStageKey
  const displayedStageKey: StageKey =
    uiStageKey === 'step_0_start' ? 'step_1_idea' : uiStageKey
  const activeSidebarIndex = getSidebarStepIndex(uiStageKey)
  const displayedActiveSidebarIndex =
    activeSidebarIndex === 0 ? 1 : activeSidebarIndex
  const currentStageExperts = getStageExperts(uiStageKey)
  const hasProcessGuideMessage = messages.some(
    (message) =>
      message.role === 'assistant' && message.content.includes('## 전체 프로세스')
  )
  const shouldShowProcessPanel =
    currentStageKey !== 'step_0_start' || hasProcessGuideMessage
  const scrollToSidebarStep = useCallback((sidebarIndex: number) => {
    const node = scrollRef.current
    if (!node) {
      return
    }

    const stageKeys = getStageKeysForSidebarIndex(sidebarIndex)
    const targetElement = stageKeys
      .map((stageKey) =>
        node.querySelector<HTMLElement>(`[data-stage-divider="${stageKey}"]`)
      )
      .find((element): element is HTMLElement => Boolean(element))

    if (targetElement) {
      targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
      return
    }

    node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' })
  }, [])
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

      const postStageTransition = (stageKey: StageKey, reason: string) =>
        fetch('/api/study/stage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            projectId,
            nextStageKey: stageKey,
            exitReason: reason,
          }),
        })

      const response = await postStageTransition(nextStageKey, exitReason)

      if (
        response.status === 409 &&
        exitReason === 'style_reference_selected' &&
        nextStageKey === 'step_5_design'
      ) {
        const conflictData = (await response.json().catch(() => null)) as {
          currentStageKey?: StageKey
        } | null
        let cursorStageKey = conflictData?.currentStageKey

        while (
          cursorStageKey &&
          cursorStageKey !== nextStageKey &&
          isKnownStageKey(cursorStageKey)
        ) {
          const intermediateStageKey = getNextStageKey(cursorStageKey)

          if (!intermediateStageKey) {
            break
          }

          const catchUpResponse = await postStageTransition(
            intermediateStageKey,
            intermediateStageKey === nextStageKey
              ? exitReason
              : 'stage_catch_up'
          )

          if (!catchUpResponse.ok) {
            const errorText = await catchUpResponse.text()
            throw new Error(
              `Stage transition failed: ${catchUpResponse.status} ${errorText}`
            )
          }

          const catchUpData = (await catchUpResponse.json()) as {
            currentStageKey?: StageKey
          }
          cursorStageKey = catchUpData.currentStageKey
        }

        if (cursorStageKey !== nextStageKey) {
          throw new Error(
            `Stage transition failed: unable to catch up from ${conflictData?.currentStageKey ?? 'unknown'} to ${nextStageKey}`
          )
        }

        setCurrentStageKey(nextStageKey)
        setPendingNextStageKey(null)
        await fetchStageTimeline()
        return
      }

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Stage transition failed: ${response.status} ${errorText}`)
      }

      const data = (await response.json()) as {
        currentStageKey?: StageKey
      }

      if (data.currentStageKey) {
        setCurrentStageKey(data.currentStageKey)
        setPendingNextStageKey(null)
        await fetchStageTimeline()
      }
    },
    [fetchStageTimeline, projectId, sessionId]
  )

  const applyStageHeaders = useCallback(
    async (response: Response, aiContent: string) => {
      const currentStageHeader = response.headers.get('x-aidee-current-stage')
      const nextStageHeader = response.headers.get('x-aidee-next-stage')
      const nextStageFromContent = getStageKeyFromProceedPrompt(aiContent)
      const nextStageCandidate =
        nextStageHeader && isKnownStageKey(nextStageHeader)
          ? nextStageHeader
          : nextStageFromContent

      if (
        nextStageCandidate &&
        nextStageCandidate !== currentStageKey &&
        isSameOrNextStage(currentStageKey, nextStageCandidate)
      ) {
        setPendingNextStageKey(nextStageCandidate)
        return
      }

      if (
        currentStageHeader &&
        isKnownStageKey(currentStageHeader) &&
        currentStageHeader !== currentStageKey &&
        isSameOrNextStage(currentStageKey, currentStageHeader)
      ) {
        setPendingNextStageKey(currentStageHeader)
        return
      }

      setPendingNextStageKey(null)
    },
    [currentStageKey]
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
        const latestRfpContentFromHistory = [...normalizedMessages]
          .reverse()
          .find(
            (message) =>
              message.normalizedMessage.role === 'assistant' &&
              (message.normalizedMessage.content.includes('# 제품 제안요청서') ||
                message.normalizedMessage.content.includes('## 1. 프로젝트 개요'))
          )?.normalizedMessage.content
        const latestImageBlockFromHistory = [...normalizedMessages]
          .reverse()
          .find((message) => message.imageBlock && message.imageBlock.purpose !== 'persona')
          ?.imageBlock

        if (latestRfpFromHistory) {
          setLatestRfpJson(latestRfpFromHistory)
        }

        if (latestRfpContentFromHistory) {
          setLatestRfpContent(latestRfpContentFromHistory)
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
        isLoading ||
        !sessionId
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
            const displayContent = stripInternalBlocksForDisplay(aiContent)
            setMessages([
              {
                id: aiMessageId,
                role: 'assistant',
                content: displayContent,
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

          if (imageBlock && imageBlock.purpose !== 'persona') {
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
      setLoadingLabelOverride(null)
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
    sessionId,
  ])

  useEffect(() => {
    const node = scrollRef.current
    if (node) {
      node.scrollTop = node.scrollHeight
    }
  }, [messages, isLoading])

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
        let errorMessage = `Project plan download failed: ${response.status}`
        const rawText = await response.text()

        try {
          const parsed = JSON.parse(rawText) as { error?: string }
          errorMessage = parsed.error
            ? `Project plan download failed: ${parsed.error}`
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
      link.download = `${projectTitle || 'aidee-project-plan'}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      console.error(error)
      alert(
        error instanceof Error
          ? error.message
          : '기획안 PDF를 생성하지 못했습니다.'
      )
    } finally {
      setIsDownloadingRfp(false)
    }
  }, [latestRfpContent, latestRfpJson, projectId, projectTitle])

  const streamAssistantResponse = async (
    nextMessages: ChatMessage[],
    stageKeyForRequest: StageKey = currentStageKey,
    expertForRequest: ExpertKey = activeExpert,
    expertCall = false,
    forceImageGeneration?: ForceImageGeneration
  ) => {
    setLoadingLabelOverride(
      isLikelyImageGenerationTurn({
        messages: nextMessages,
        stageKey: stageKeyForRequest,
      })
        ? '이미지를 생성하고 있어요.'
        : null
    )

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: buildChatApiMessages(nextMessages),
        projectId,
        currentStageKey: stageKeyForRequest,
        activeExpert: expertForRequest,
        expertCall,
        forceImageGeneration,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      const aiMessageId = crypto.randomUUID()
      const createdAt = new Date().toISOString()
      setMessages((prev) => [
        ...prev,
        {
          id: aiMessageId,
          role: 'assistant',
          content:
            errorText ||
            '이미지 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
          active_agent: expertForRequest,
          created_at: createdAt,
          stage_key: stageKeyForRequest,
        },
      ])
      throw new Error(`Chat request failed: ${response.status} ${errorText}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const responseStageKey = getResponseStageKey(response, stageKeyForRequest)
    const decoder = new TextDecoder()
    let aiContent = ''
    const aiMessageId = crypto.randomUUID()
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
      const displayContent = stripInternalBlocksForDisplay(aiContent)
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId ? { ...msg, content: displayContent } : msg
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

    if (imageBlock && imageBlock.purpose !== 'persona') {
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
  setLoadingLabelOverride(null)
  return normalizedMessage
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
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const requestNextStage = async (targetStageKey?: StageKey | null) => {
    if (isLoading) {
      return
    }

    const candidateStageKey =
      targetStageKey ?? pendingNextStageKey ?? getNextStageKey(currentStageKey)
    const nextStageKey =
      candidateStageKey && isSameOrNextStage(currentStageKey, candidateStageKey)
        ? candidateStageKey
        : getNextStageKey(currentStageKey)

    if (!nextStageKey) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: '다음 단계',
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

      await transitionStage(nextStageKey, 'manual_advance')
      setPendingNextStageKey(null)
      await streamAssistantResponse(nextMessages, nextStageKey, 'aidee')
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const getIntentStageKey = (_text: string, fallbackStageKey: StageKey) => {
    void _text
    return fallbackStageKey
  }

  const onFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || !sessionId) {
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
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const sendChatAction = async (actionText: string) => {
    if (!actionText.trim() || isLoading || !sessionId) {
      return
    }

    const stageKeyForRequest = getIntentStageKey(actionText, currentStageKey)
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: actionText,
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: stageKeyForRequest,
    }
    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setIsExpertPickerOpen(false)
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: actionText,
        activeAgent: 'aidee',
      })

      await streamAssistantResponse(nextMessages, stageKeyForRequest, 'aidee')
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const handleMoreToSay = async () => {
    if (isLoading || !sessionId) {
      return
    }

    const createdAt = new Date().toISOString()
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: '더 하고 싶은 말이 있어요',
      active_agent: 'aidee',
      created_at: createdAt,
      stage_key: currentStageKey,
    }
    const assistantMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content:
        '좋아요. 이 단계에서 더 반영하고 싶은 내용을 편하게 적어주세요. 말씀해주신 내용을 기준에 함께 반영하겠습니다.',
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }

    setPendingNextStageKey(null)
    setMessages((prev) => [...prev, userMessage, assistantMessage])

    try {
      await insertMessage({
        role: 'user',
        content: userMessage.content,
        activeAgent: 'aidee',
      })
      await insertMessage({
        role: 'assistant',
        content: assistantMessage.content,
        activeAgent: 'aidee',
      })
    } catch (error) {
      console.error(error)
    }
  }

  const buildClientStageTransitionPrompt = (nextStageKey: StageKey) => {
    const step = getProcessStepForStage(nextStageKey)

    if (nextStageKey === 'step_3_direction') {
      return [
        '<<AIDEE_DIRECTION_WIDGETS>>',
        '<</AIDEE_DIRECTION_WIDGETS>>',
        '',
        `STEP ${step.index}. ${step.title} 단계로 넘어왔습니다.`,
        '화면의 세 가지 위젯 중 궁금한 항목을 눌러주세요.',
      ].join('\n')
    }

    return [
      `다음으로 STEP ${step.index}. ${step.title} 단계로 넘어가겠습니다.`,
      `이 단계에서는 ${step.description}`,
      '진행할까요?',
    ].join('\n')
  }

  const confirmPersonaCard = async (messageId: string) => {
    if (isLoading || !sessionId) {
      return
    }

    const nextStageKey: StageKey = 'step_3_direction'
    const now = new Date().toISOString()
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: '페르소나 카드를 확정합니다.',
      active_agent: 'aidee',
      created_at: now,
      stage_key: currentStageKey,
    }
    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: buildClientStageTransitionPrompt(nextStageKey),
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: nextStageKey,
    }

    setConfirmedPersonaMessageIds((prev) => ({
      ...prev,
      [messageId]: true,
    }))
    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: userMessage.content,
        activeAgent: 'aidee',
      })
      await transitionStage(nextStageKey, 'persona_card_confirmed')
      await insertMessage({
        role: 'assistant',
        content: assistantMessage.content,
        activeAgent: 'aidee',
      })
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const confirmPersonaFlowCard = async (
    messageId: string,
    kind: Exclude<PersonaArtifactKind, 'persona'>
  ) => {
    if (kind !== 'problem_statements' || isLoading || !sessionId) {
      return
    }

    setConfirmedPersonaFlowCardIds((prev) => ({
      ...prev,
      [messageId]: true,
    }))
    await sendChatAction('Problem Statements 카드를 확정합니다.')
  }

  const confirmVisualizationMessage = (messageId: string) => {
    setConfirmedVisualizationMessageIds((prev) => ({
      ...prev,
      [messageId]: true,
    }))
  }

  const requestVisualizationRevision = async (label: string) => {
    if (isLoading) {
      return
    }

    await sendChatAction(`수정하기: ${label} 내용을 수정하고 싶어요.`)
  }

  const visualizePersonaArtifact = async (
    messageId: string,
    personaContent: string
  ) => {
    if (isLoading || !sessionId || !confirmedVisualizationMessageIds[messageId]) {
      return
    }

    const artifactKind = getPersonaArtifactKind(personaContent)
    if (!artifactKind) {
      return
    }
    const forceImageGeneration = getPersonaArtifactForce(artifactKind)
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: ['시각화하기', personaContent].join('\n\n'),
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }
    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setLoadingLabelOverride(
      artifactKind === 'persona'
        ? '페르소나를 시각화하고 있어요.'
        : '카드를 만들고 있어요.'
    )
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: '시각화하기',
        activeAgent: 'aidee',
      })

      const responseMessage = await streamAssistantResponse(
        nextMessages,
        currentStageKey,
        'aidee',
        false,
        forceImageGeneration
      )

      if (
        responseMessage?.generatedImages?.length ||
        extractPersonaFlowCard(responseMessage?.content ?? '')
      ) {
        setVisualizedPersonaMessageIds((prev) => ({
          ...prev,
          [messageId]: true,
        }))
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const visualizeDirectionArtifact = async (
    messageId: string,
    directionContent: string,
    directionKind: DirectionArtifactKind
  ) => {
    if (isLoading || !sessionId) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: ['시각화하기', directionContent].join('\n\n'),
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }
    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setLoadingLabelOverride('카드를 만들고 있어요.')
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: '시각화하기',
        activeAgent: 'aidee',
      })

      const responseMessage = await streamAssistantResponse(
        nextMessages,
        currentStageKey,
        'aidee',
        false,
        getDirectionArtifactForce(directionKind)
      )

      if (extractDirectionCard(responseMessage?.content ?? '')) {
        setVisualizedDirectionMessageIds((prev) => ({
          ...prev,
          [messageId]: true,
        }))
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const visualizeStyleMoodboard = async (
    messageId: string,
    styleContent: string
  ) => {
    if (isLoading || !sessionId || !confirmedVisualizationMessageIds[messageId]) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: ['시각화하기', styleContent].join('\n\n'),
      active_agent: 'aidee',
      created_at: new Date().toISOString(),
      stage_key: currentStageKey,
    }
    const nextMessages = [...messages, userMessage]

    setMessages(nextMessages)
    setLoadingLabelOverride('무드보드를 생성하고 있어요.')
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: '시각화하기',
        activeAgent: 'aidee',
      })

      await streamAssistantResponse(
        nextMessages,
        currentStageKey,
        'aidee',
        false,
        'style_moodboard_visualization'
      )

      setVisualizedDirectionMessageIds((prev) => ({
        ...prev,
        [messageId]: true,
      }))
    } catch (error) {
      console.error(error)
    } finally {
      setLoadingLabelOverride(null)
      setIsLoading(false)
    }
  }

  const sendGeneratedImageSelection = async (
    messageId: string,
    imageIndex: number
  ) => {
    if (isLoading) {
      return
    }

    const actionText = [
      `스타일 레퍼런스 ${imageIndex + 1}번을 선택합니다.`,
      '선택한 스타일 레퍼런스를 텍스트로 정리해주세요.',
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
    setLoadingLabelOverride('이미지를 생성하고 있어요.')
    setIsLoading(true)

    try {
      await insertMessage({
        role: 'user',
        content: actionText,
        activeAgent: 'aidee',
      })

      await streamAssistantResponse(
        nextMessages,
        currentStageKey,
        'aidee'
      )
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
      setLoadingLabelOverride(null)
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

  const hintModalMessage = hintModalMessageId
    ? messages.find((message) => message.id === hintModalMessageId)
    : null
  const hintModalChoiceSplit =
    hintModalMessage?.role === 'assistant'
      ? splitAssistantChoices(hintModalMessage.content)
      : null
  const hintModalChoices =
    hintModalMessage && hintModalChoiceSplit
      ? hintModalChoiceSplit.choices
      : []

  return (
    <div className="flex h-screen w-full overflow-hidden bg-white">
      <aside className="flex w-64 shrink-0 justify-center border-r border-gray-200 bg-neutral-50 p-2">
        <div className="flex h-full flex-1 flex-col justify-between">
          <div className="flex flex-col gap-8">
            <div className="flex w-60 items-center justify-between">
              <Link href="/" className="inline-block transition-opacity hover:opacity-80">
                <Image
                  src="/brand/aidee-logo-blue.svg"
                  alt="Aidee"
                  width={115}
                  height={40}
                  unoptimized
                  priority
                  className="h-7 w-auto"
                />
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

            {shouldShowProcessPanel ? (
              <div className="flex w-60 flex-col gap-1">
                <div className="text-xs leading-5 font-medium text-slate-500">
                  디자인 프로세스
                </div>
                <div className="inline-flex items-start gap-1.5">
                  <div className="inline-flex w-2.5 flex-col items-start">
                    {SIDEBAR_STEPS.map((_, index) => {
                      const stepNumber = index + 1
                      const isActive = stepNumber === displayedActiveSidebarIndex
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
                              isActive
                                ? 'border-2 border-blue-600 bg-white'
                                : 'bg-gray-200'
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
                      const stepNumber = index + 1
                      const isActive = stepNumber === displayedActiveSidebarIndex
                      return (
                        <button
                          key={step}
                          type="button"
                          onClick={() => scrollToSidebarStep(stepNumber)}
                          className={`inline-flex self-stretch items-center gap-2 rounded-xl py-1.5 text-left transition ${
                            isActive
                              ? 'bg-blue-50/70'
                              : 'hover:bg-gray-50'
                          }`}
                        >
                          <div
                            className={`text-sm leading-6 font-medium ${
                              isActive ? 'text-blue-600' : 'text-gray-300'
                            }`}
                          >
                            {stepNumber}. {step}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </div>
            ) : null}

            <div className="flex w-60 flex-col gap-2">
              <div className="text-xs leading-4 font-medium text-slate-500">
                AI 전문가
              </div>
              <div className="flex w-60 flex-col gap-1">
                {EXPERT_DEFINITIONS.filter((expert) => expert.key !== 'aidee').map(
                  (expert) => (
                    <SidebarExpertRow
                      key={expert.key}
                      expertKey={expert.key}
                      selected={currentStageExperts.includes(expert.key)}
                    />
                  )
                )}
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
              <a
                href="https://forms.gle/fFmtcX7DMh27pcqMA"
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-2 rounded-[999px] bg-gradient-to-bl from-blue-600/0 to-blue-600/40 px-4 py-1 transition hover:from-blue-600/10 hover:to-blue-600/60 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                <div className="h-4 w-4 overflow-hidden">
                  <div className="h-3.5 w-3.5 outline outline-[1.5px] outline-offset-[-0.75px] outline-white" />
                </div>
                <div className="text-sm leading-5 font-medium text-white">
                  Basic
                </div>
              </a>
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
              <p className="text-xs text-zinc-400">
                현재 단계: {getStageLabel(displayedStageKey)}
              </p>
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
                새로운 프로젝트가 시작되었네요! &apos;Aidee&apos;팀과 함께 아이디어를 구체화해보아요.
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

            const projectDirectionCard =
              m.role === 'assistant' ? extractProjectDirectionCard(m.content) : null
            const personaFlowCard =
              m.role === 'assistant' ? extractPersonaFlowCard(m.content) : null
            const directionWidgets =
              m.role === 'assistant' ? hasDirectionWidgets(m.content) : false
            const directionCard =
              m.role === 'assistant' ? extractDirectionCard(m.content) : null
            const styleKeywordPicker =
              m.role === 'assistant' ? hasStyleKeywordPicker(m.content) : false
            const assistantBaseContent =
              m.role === 'assistant'
                ? stripStyleKeywordPicker(
                    stripDirectionInternalBlocks(
                      personaFlowCard
                        ? stripPersonaFlowCard(m.content)
                        : m.content
                    )
                  )
                : m.content
            const projectDirectionDisplayParts =
              m.role === 'assistant' && projectDirectionCard
                ? splitProjectDirectionContent(assistantBaseContent)
                : null
            const contentForDisplay =
              m.role === 'assistant'
                ? projectDirectionDisplayParts
                  ? projectDirectionDisplayParts.after
                  : stripProjectDirectionCard(assistantBaseContent)
                : m.content
            const personaArtifactKind =
              m.role === 'assistant' ? getPersonaArtifactKind(m.content) : null
            const directionResearchKind =
              m.role === 'assistant' ? getDirectionResearchKind(m.content) : null
            const isStyleProposal =
              m.role === 'assistant' && isStyleReferenceProposal(m.content)
            const visualizationLabel =
              personaArtifactKind === 'problem_statements'
                ? 'Problem Statements'
                : personaArtifactKind === 'experience_keywords'
                  ? 'Keywords: Experience'
                  : personaArtifactKind === 'relationship_keywords'
                    ? 'Keywords: Relationship'
                    : personaArtifactKind === 'persona'
                      ? 'Persona Summary'
                      : directionResearchKind === 'market_size'
                        ? '시장 규모 리서치'
                        : directionResearchKind === 'consumption_keywords'
                          ? '소비 트렌드 리서치'
                          : directionResearchKind === 'brand_positioning'
                            ? '경쟁사 리서치'
                            : isStyleProposal
                              ? '스타일 레퍼런스'
                              : null
            const personaSummaryData =
              m.role === 'assistant' ? parsePersonaData(m.content) : null
            const isPersonaSummary =
              m.role === 'assistant' &&
              stageKeyForMessage === 'step_2_persona' &&
              !personaFlowCard &&
              (isPersonaSummaryText(m.content) || Boolean(personaSummaryData))

            const isRfpMessage =
              m.role === 'assistant' &&
              (m.content.includes('# 제품 제안요청서') ||
                m.content.includes('## 1. 프로젝트 개요'))
            const isStyleReferenceImageMessage =
              m.generatedImagePurpose === 'style_reference'
            const isPersonaVisualizationMessage =
              m.role === 'assistant' &&
              m.generatedImagePurpose === 'persona' &&
              Boolean(m.generatedImages?.length)
            const personaVisualizationSourceData = isPersonaVisualizationMessage
              ? messages
                  .slice(0, index)
                  .reverse()
                  .filter(
                    (message) =>
                      message.role === 'assistant' &&
                      message.stage_key === 'step_2_persona'
                  )
                  .map((message) => parsePersonaVisualData(message.content))
                  .find((data) => Boolean(data))
              : null
            const assistantChoiceSplit =
              m.role === 'assistant'
                ? splitAssistantChoices(contentForDisplay)
                : { displayContent: m.content, choices: [] }
            const showProcessConfirmButton =
              m.role === 'assistant' &&
              m.id === latestAideeAssistantId &&
              (!m.active_agent || m.active_agent === 'aidee') &&
              m.content.includes('## 전체 내용 정리') &&
              m.content.includes('프로세스 확인하기') &&
              !hasProcessGuideMessage
            const stageProceedNextStageKey = getStageKeyFromProceedPrompt(
              m.content
            )
            const hasStageProceedPrompt =
              m.role === 'assistant' && isStageProceedPrompt(m.content)
            const fallbackProceedStageKey =
              hasStageProceedPrompt && !stageProceedNextStageKey
                ? getNextStageKey(stageKeyForMessage)
                : null
            const promptProceedStageKey =
              stageProceedNextStageKey &&
              stageProceedNextStageKey !== currentStageKey &&
              isSameOrNextStage(stageKeyForMessage, stageProceedNextStageKey)
                ? stageProceedNextStageKey
                : null
            const proceedButtonStageKey =
              pendingNextStageKey ?? promptProceedStageKey ?? fallbackProceedStageKey
            const showStageProceedButtons = Boolean(
              m.role === 'assistant' &&
                m.id === latestAideeAssistantId &&
                (!m.active_agent || m.active_agent === 'aidee') &&
                proceedButtonStageKey &&
                hasStageProceedPrompt
            )
            const showHintButton =
              m.role === 'assistant' &&
              (!m.active_agent || m.active_agent === 'aidee') &&
              contentForDisplay.trim() &&
              !showStageProceedButtons &&
              !hasStageProceedPrompt &&
              !showProcessConfirmButton &&
              !isPersonaSummary &&
              !styleKeywordPicker &&
              !isStyleProposal &&
              assistantChoiceSplit.choices.length > 0
            const personaArtifactWasVisualized = Boolean(
              personaArtifactKind &&
                (visualizedPersonaMessageIds[m.id] ||
                  messages.slice(index + 1).some((message) => {
                    if (personaArtifactKind === 'persona') {
                      return (
                        message.role === 'assistant' &&
                        message.generatedImagePurpose === 'persona'
                      )
                    }

                    const laterCard = extractPersonaFlowCard(message.content)
                    return laterCard?.kind === personaArtifactKind
                  }))
            )
            const directionResearchWasVisualized = Boolean(
              directionResearchKind &&
                (visualizedDirectionMessageIds[m.id] ||
                  messages.slice(index + 1).some((message) => {
                    const laterCard = extractDirectionCard(message.content)
                    return laterCard?.kind === directionResearchKind
                  }))
            )
            const styleMoodboardWasVisualized = Boolean(
              isStyleProposal && visualizedDirectionMessageIds[m.id]
            )
            const canVisualizePersonaArtifact = Boolean(
              isPersonaSummary &&
                personaArtifactKind &&
                !personaArtifactWasVisualized
            )
            const canVisualizeDirectionArtifact = Boolean(
              directionResearchKind && !directionResearchWasVisualized
            )
            const canVisualizeStyleMoodboard = Boolean(
              isStyleProposal && !styleMoodboardWasVisualized
            )
            const showVisualizationGate = Boolean(
              visualizationLabel &&
                (canVisualizePersonaArtifact ||
                  canVisualizeDirectionArtifact ||
                  canVisualizeStyleMoodboard)
            )
            const visualizationMessageConfirmed = Boolean(
              confirmedVisualizationMessageIds[m.id]
            )
            const personaCardWasConfirmed = Boolean(
              confirmedPersonaMessageIds[m.id] ||
                messages.slice(index + 1).some(
                  (message) =>
                    message.role === 'user' &&
                    message.content.includes('페르소나 카드를 확정합니다')
                )
            )
            const personaFlowCardWasSuperseded = Boolean(
              personaFlowCard &&
                messages.slice(index + 1).some((message) => {
                  const laterCard = extractPersonaFlowCard(message.content)
                  return laterCard?.kind === personaFlowCard.kind
                })
            )
            const personaFlowCardWasConfirmed = Boolean(
              personaFlowCard &&
                (confirmedPersonaFlowCardIds[m.id] ||
                  messages.slice(index + 1).some(
                    (message) =>
                      message.role === 'user' &&
                      message.content.includes(
                        `${personaFlowCard.kind === 'problem_statements' ? 'Problem Statements' : ''} 카드를 확정합니다`
                      )
                  ))
            )
            const showProblemStatementsCardActions = Boolean(
              personaFlowCard?.kind === 'problem_statements' &&
                !personaFlowCardWasConfirmed &&
                !personaFlowCardWasSuperseded
            )

            if (
              isPersonaVisualizationMessage &&
              personaVisualizationSourceData &&
              m.generatedImages?.[0]
            ) {
              return (
                <div key={m.id} className="space-y-6">
                  {shouldShowStageDivider ? (
                    <StageDivider stageKey={stageKeyForMessage} />
                  ) : null}
                  <div className="flex flex-col items-start">
                    <PersonaCard
                      data={{
                        ...personaVisualizationSourceData,
                        imageUrl: m.generatedImages[0],
                      }}
                      showActions={false}
                    />
                    <div className="max-w-[514px] min-w-0 overflow-hidden rounded-[24px] rounded-tl-none bg-gray-200 p-5 text-base font-medium leading-relaxed text-neutral-900 shadow-sm">
                      <div className="prose prose-sm prose-p:my-0 prose-p:leading-7 max-w-full break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkBreaks]}
                        >
                          {assistantChoiceSplit.displayContent}
                        </ReactMarkdown>
                      </div>
                    </div>
                    {!personaCardWasConfirmed ? (
                      <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() => void confirmPersonaCard(m.id)}
                          className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          확정하기
                        </button>
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() =>
                            void sendChatAction(
                              '수정하기: 페르소나 카드 내용을 조금 더 조정하고 싶어요.'
                            )
                          }
                          className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          수정하기
                        </button>
                      </div>
                    ) : null}
                    {showStageProceedButtons ? (
                      <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() =>
                            void requestNextStage(proceedButtonStageKey)
                          }
                          className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          다음 단계로
                        </button>
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() => void handleMoreToSay()}
                          className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          더 하고 싶은 말이 있어요
                        </button>
                      </div>
                    ) : null}
                  </div>
                </div>
              )
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
                  {projectDirectionDisplayParts?.before ? (
                    <div className="max-w-[514px] min-w-0 overflow-hidden rounded-[24px] rounded-tl-none bg-gray-200 p-5 text-base font-medium leading-relaxed text-neutral-900 shadow-sm">
                      <div className="prose prose-sm prose-p:my-0 prose-p:leading-7 max-w-full break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkBreaks]}
                        >
                          {projectDirectionDisplayParts.before}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : null}
                  {projectDirectionCard ? (
                    <ProjectDirectionCard data={projectDirectionCard} />
                  ) : null}
                  {personaFlowCard ? (
                    <PersonaFlowCard
                      kind={personaFlowCard.kind}
                      summary={personaFlowCard.summary}
                    />
                  ) : null}
                  {showProblemStatementsCardActions && personaFlowCard ? (
                    <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                      <button
                        type="button"
                        disabled={isLoading}
                        onClick={() =>
                          void sendChatAction(
                            'Problem Statements 카드를 수정하고 싶어요.'
                          )
                        }
                        className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        수정하기
                      </button>
                      <button
                        type="button"
                        disabled={isLoading}
                        onClick={() =>
                          void confirmPersonaFlowCard(m.id, personaFlowCard.kind)
                        }
                        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        확정하기
                      </button>
                    </div>
                  ) : null}
                  {directionWidgets ? (
                    <DirectionResearchWidgets
                      disabled={isLoading}
                      onSelect={(kind) => {
                        const labelMap: Record<DirectionArtifactKind, string> = {
                          market_size: '시장 규모 리서치 보기',
                          consumption_keywords: '소비 트렌드 리서치 보기',
                          brand_positioning: '경쟁사 리서치 보기',
                        }

                        void sendChatAction(labelMap[kind])
                      }}
                    />
                  ) : null}
                  {directionCard ? (
                    <DirectionResearchCard
                      kind={directionCard.kind}
                      summary={directionCard.summary}
                    />
                  ) : null}
                  {styleKeywordPicker ? (
                    <StyleKeywordPicker
                      disabled={isLoading}
                      onSubmit={(text) => void sendChatAction(text)}
                    />
                  ) : null}
                  {contentForDisplay.trim() ? (
                    <div
                      className={`max-w-[514px] min-w-0 overflow-hidden rounded-[24px] p-5 text-base leading-relaxed font-medium shadow-sm ${
                        m.role === 'user'
                          ? 'rounded-tr-none bg-gray-100 text-neutral-900'
                          : 'rounded-tl-none bg-gray-200 text-neutral-900'
                      }`}
                    >
                      <div className="prose prose-sm prose-p:my-0 prose-p:leading-7 prose-li:my-0 prose-headings:mb-3 prose-strong:text-neutral-900 max-w-full break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkBreaks]}
                          components={{
                            p: ({ children }) => (
                              <p className="mb-1 max-w-full break-words leading-7 last:mb-0">
                                {children}
                              </p>
                            ),
                            ul: ({ children }) => (
                              <ul className="my-3 max-w-full list-disc space-y-1 pl-5">
                                {children}
                              </ul>
                            ),
                            ol: ({ children }) => (
                              <ol className="my-3 max-w-full list-decimal space-y-1 pl-5">
                                {children}
                              </ol>
                            ),
                            li: ({ children }) => (
                              <li className="max-w-full break-words leading-5 [&>p]:mb-0 [&>p]:inline">
                                {children}
                              </li>
                            ),
                            br: () => <br className="block h-1" />,
                          }}
                        >
                          {assistantChoiceSplit.displayContent}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : null}
                {showHintButton ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => setHintModalMessageId(m.id)}
                      className="rounded-full bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-blue-50 hover:text-blue-700 hover:outline-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      힌트보기
                    </button>
                  </div>
                ) : null}
                {showProcessConfirmButton ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => void sendChatAction('프로세스 확인하기')}
                      className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      프로세스 확인하기
                    </button>
                  </div>
                ) : null}
                {showStageProceedButtons ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() =>
                        void requestNextStage(proceedButtonStageKey)
                      }
                      className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      다음 단계로
                    </button>
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => void handleMoreToSay()}
                      className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      더 하고 싶은 말이 있어요
                    </button>
                  </div>
                ) : null}
                {showVisualizationGate &&
                visualizationLabel &&
                !canVisualizeDirectionArtifact ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() =>
                        void requestVisualizationRevision(visualizationLabel)
                      }
                      className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm outline outline-1 outline-gray-200 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      수정하기
                    </button>
                    <button
                      type="button"
                      disabled={isLoading || visualizationMessageConfirmed}
                      onClick={() => confirmVisualizationMessage(m.id)}
                      className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {visualizationMessageConfirmed ? '확정됨' : '확정하기'}
                    </button>
                  </div>
                ) : null}
                {isPersonaSummary && personaArtifactKind ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    {!personaArtifactWasVisualized &&
                    visualizationMessageConfirmed ? (
                      <button
                        type="button"
                        disabled={isLoading}
                        onClick={() =>
                          void visualizePersonaArtifact(m.id, m.content)
                        }
                        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        시각화하기
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {directionResearchKind ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    {!directionResearchWasVisualized ? (
                      <button
                        type="button"
                        disabled={isLoading}
                        onClick={() =>
                          void visualizeDirectionArtifact(
                            m.id,
                            m.content,
                            directionResearchKind
                          )
                        }
                        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        시각화하기
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {isStyleProposal ? (
                  <div className="mt-2 flex max-w-[602px] flex-wrap gap-2">
                    {!styleMoodboardWasVisualized &&
                    visualizationMessageConfirmed ? (
                      <button
                        type="button"
                        disabled={isLoading}
                        onClick={() => void visualizeStyleMoodboard(m.id, m.content)}
                        className="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        시각화하기
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {m.role === 'assistant' && m.generatedImages?.length ? (
                  <div className="mt-3 w-full max-w-[760px] space-y-2">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-medium text-slate-500">
                        생성된 이미지 {m.generatedImages.length}장
                      </p>
                    </div>
                    <div
                      className={`grid gap-3 ${
                        isStyleReferenceImageMessage &&
                        m.generatedImages.length === 3
                          ? 'grid-cols-1 sm:grid-cols-3'
                          : 'grid-cols-1 sm:grid-cols-2'
                      }`}
                    >
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
                            className={`w-full object-cover ${
                              isStyleReferenceImageMessage
                                ? 'h-40'
                                : 'aspect-[4/3]'
                            }`}
                          />
                          {isStyleReferenceImageMessage ? (
                            <div className="flex items-center justify-between gap-3 border-t border-slate-100 px-3 py-2">
                              <span className="text-xs font-medium text-slate-500">
                                스타일 {index + 1}
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
                                    index
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
                                  : '이 스타일 선택'}
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
                  />
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
                          className="ml-5 min-w-0 max-w-[602px] overflow-hidden rounded-2xl bg-gray-100 p-3 text-sm font-medium leading-6 text-neutral-800"
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
                            <div className="prose prose-sm prose-p:my-0 prose-p:leading-6 prose-li:my-0 max-w-full break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkBreaks]}
                                components={{
                                  p: ({ children }) => (
                                    <p className="mb-1 max-w-full break-words leading-6 last:mb-0">
                                      {children}
                                    </p>
                                  ),
                                  ul: ({ children }) => (
                                    <ul className="my-2 max-w-full list-disc space-y-1 pl-5">
                                      {children}
                                    </ul>
                                  ),
                                  ol: ({ children }) => (
                                    <ol className="my-2 max-w-full list-decimal space-y-1 pl-5">
                                      {children}
                                    </ol>
                                  ),
                                  li: ({ children }) => (
                                    <li className="max-w-full break-words leading-5 [&>p]:mb-0 [&>p]:inline">
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

          {(currentStageKey === 'step_6_company' ||
            pendingNextStageKey === 'step_6_company') &&
          latestRfpContent ? (
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
                {loadingLabelOverride ?? activeExpertDefinition.loadingLabel}
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
                className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-3 text-base leading-relaxed font-medium text-neutral-900 caret-blue-600 outline-none placeholder:text-zinc-400"
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
      {hintModalMessage && hintModalChoiceSplit ? (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/20 px-4 py-6 sm:items-center">
          <div className="w-full max-w-sm rounded-2xl bg-white p-4 shadow-2xl">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-neutral-900">
                  답변 힌트
                </h3>
                <p className="mt-1 text-xs font-medium leading-5 text-slate-400">
                  프로젝트 맥락을 반영한 작성 방향입니다. 그대로 고르거나 직접 바꿔 입력해도 됩니다.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setHintModalMessageId(null)}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-100 text-sm font-semibold text-slate-500 transition hover:bg-slate-200"
                aria-label="힌트 닫기"
              >
                ×
              </button>
            </div>

            <div className="mt-4 flex flex-col gap-2">
              {hintModalChoices.map((choice) => (
                <button
                  key={`${hintModalMessage.id}-${choice.key}-${choice.label}`}
                  type="button"
                  disabled={isLoading}
                  onClick={() => {
                    setHintModalMessageId(null)
                    void sendChatAction(choice.value)
                  }}
                  className="w-full rounded-xl bg-slate-50 px-3 py-2.5 text-left text-sm font-semibold leading-5 text-slate-700 break-words transition [overflow-wrap:anywhere] hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {choice.key}. {choice.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
