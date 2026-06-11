import { z } from 'zod'

export const RFP_DOCUMENT_SCHEMA = z.object({
  projectName: z.string(),
  oneLineDefinition: z.string(),
  projectGoal: z.string(),
  finalPurpose: z.string(),
  mainTarget: z.string(),
  usageContext: z.string(),
  coreNeeds: z.string(),
  coreValue: z.string(),
  styleKeywords: z.array(z.string()).min(1),
  avoidDirections: z.array(z.string()).min(1),
  mustHaveFeatures: z.array(z.string()).min(1),
  niceToHaveFeatures: z.array(z.string()),
  excludedFeatures: z.array(z.string()).min(1),
  budgetRange: z.string(),
  timeline: z.string(),
  sizeOrForm: z.string(),
  implementationNotes: z.array(z.string()).min(1),
  referenceSummary: z.string(),
  researchInsights: z.array(z.string()).min(1),
  successCriteria: z.array(z.string()).min(1),
  nextActions: z.array(z.string()).min(1),
})

export type RfpDocument = z.infer<typeof RFP_DOCUMENT_SCHEMA>

export function buildRfpObjectPrompt({
  projectTitle,
  requirements,
  referenceContext,
  conversation,
}: {
  projectTitle: string
  requirements: string
  referenceContext: string
  conversation: string
}) {
  return `
당신은 Aidee의 STEP 6 프로젝트 기획안 생성 담당자입니다.
반드시 한국어로, 주어진 프로젝트 컨텍스트와 대화 기록만 바탕으로 구조화된 프로젝트 기획안 JSON을 생성하세요.
없는 정보를 지어내지 말고, 추상적인 표현은 구체적인 실행 문장으로 바꾸세요.
가능한 경우 STEP 1~4에서 확정된 내용만 사용하고, 불확실한 항목은 "미정"이 아니라 현재 대화에서 가장 방어 가능한 표현으로 정리하세요.

[프로젝트명]
${projectTitle}

[프로젝트 requirements]
${requirements}

[레퍼런스 이미지 분석]
${referenceContext}

[대화 기록]
${conversation}

[작성 규칙]
- oneLineDefinition은 한 문장
- styleKeywords는 3개 내외
- mustHaveFeatures / excludedFeatures / nextActions는 bullet로 바로 쓸 수 있을 정도로 구체적
- implementationNotes는 실제 제작/개발 시 주의점
- successCriteria는 완료 판단 기준
- 레퍼런스 이미지 분석에서 moodKeywords, colorKeywords, materialKeywords, shapeKeywords, detailPoints, designDirection이 보이면 그것을 실제 프로젝트 기획안 기준으로 우선 반영
- referenceSummary와 스타일/기능 방향은 레퍼런스 이미지의 핵심 무드, 재질, 형태, 디테일을 최대한 보존하는 방향으로 작성
- 출력은 스키마에 맞는 JSON 객체만 생성
`.trim()
}

export function extractRfpJsonBlock(text: string) {
  const match = text.match(
    /<<AIDEE_RFP_JSON>>[\s\n]*([\s\S]*?)[\s\n]*<<\/AIDEE_RFP_JSON>>/
  )

  if (!match) {
    return {
      cleanedText: text.trim(),
      rfpJson: null as RfpDocument | null,
    }
  }

  const cleanedText = text.replace(
    /\n?<<AIDEE_RFP_JSON>>[\s\S]*?<<\/AIDEE_RFP_JSON>>\s*$/,
    ''
  )

  try {
    const parsed = JSON.parse(match[1]) as unknown
    const validated = RFP_DOCUMENT_SCHEMA.parse(parsed)
    return {
      cleanedText: cleanedText.trim(),
      rfpJson: validated,
    }
  } catch {
    return {
      cleanedText: cleanedText.trim(),
      rfpJson: null as RfpDocument | null,
    }
  }
}
