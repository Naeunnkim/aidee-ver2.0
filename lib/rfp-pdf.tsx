import { resolve } from 'node:path'

import {
  Document,
  Font,
  Image,
  Page,
  StyleSheet,
  Text,
  View,
} from '@react-pdf/renderer'

import { type RfpDocument } from '@/lib/rfp'

const notoSansKrPath = resolve(
  process.cwd(),
  'node_modules/@fontsource/noto-sans-kr/files/noto-sans-kr-korean-400-normal.woff'
)

Font.register({
  family: 'NotoSansKR',
  src: notoSansKrPath,
})

const styles = StyleSheet.create({
  page: {
    paddingTop: 36,
    paddingBottom: 42,
    paddingHorizontal: 36,
    fontFamily: 'NotoSansKR',
    fontSize: 11,
    color: '#111827',
    lineHeight: 1.5,
  },
  header: {
    marginBottom: 18,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  title: {
    fontSize: 22,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 10,
    color: '#4B5563',
  },
  section: {
    marginTop: 14,
  },
  sectionTitle: {
    fontSize: 14,
    marginBottom: 8,
    color: '#1D4ED8',
  },
  row: {
    marginBottom: 5,
  },
  label: {
    fontSize: 10,
    color: '#374151',
    marginBottom: 2,
  },
  value: {
    fontSize: 11,
    color: '#111827',
  },
  list: {
    marginTop: 3,
  },
  bullet: {
    marginBottom: 3,
    paddingLeft: 10,
  },
  onePage: {
    paddingTop: 24,
    paddingBottom: 24,
    paddingHorizontal: 26,
    fontFamily: 'NotoSansKR',
    fontSize: 8.5,
    color: '#111827',
    lineHeight: 1.35,
  },
  onePageHeader: {
    marginBottom: 10,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#D1D5DB',
  },
  onePageTitle: {
    fontSize: 18,
    marginBottom: 4,
  },
  onePageSubtitle: {
    fontSize: 9,
    color: '#4B5563',
  },
  onePageGrid: {
    display: 'flex',
    flexDirection: 'row',
    gap: 12,
  },
  onePageLeft: {
    width: '38%',
  },
  onePageRight: {
    width: '62%',
  },
  heroImage: {
    width: '100%',
    height: 210,
    objectFit: 'cover',
    borderRadius: 10,
    marginBottom: 8,
  },
  imagePlaceholder: {
    width: '100%',
    height: 210,
    borderRadius: 10,
    backgroundColor: '#F3F4F6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  compactSection: {
    marginBottom: 8,
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#F9FAFB',
  },
  compactSectionTitle: {
    fontSize: 10,
    color: '#1D4ED8',
    marginBottom: 4,
  },
  compactLabel: {
    fontSize: 7.5,
    color: '#6B7280',
    marginBottom: 1,
  },
  compactValue: {
    fontSize: 8.5,
    color: '#111827',
    marginBottom: 3,
  },
  tagRow: {
    display: 'flex',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: 3,
  },
  tag: {
    paddingVertical: 2,
    paddingHorizontal: 5,
    borderRadius: 999,
    backgroundColor: '#E0F2FE',
    color: '#075985',
    fontSize: 7.5,
  },
})

function BulletList({ items }: { items: string[] }) {
  return (
    <View style={styles.list}>
      {items.map((item, index) => (
        <Text key={`${item}-${index}`} style={styles.bullet}>
          • {item}
        </Text>
      ))}
    </View>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.value}>{value}</Text>
    </View>
  )
}

function CompactField({ label, value }: { label: string; value: string }) {
  return (
    <View>
      <Text style={styles.compactLabel}>{label}</Text>
      <Text style={styles.compactValue}>{value}</Text>
    </View>
  )
}

function CompactBulletList({ items }: { items: string[] }) {
  return (
    <View>
      {items.slice(0, 4).map((item, index) => (
        <Text key={`${item}-${index}`} style={styles.compactValue}>
          • {item}
        </Text>
      ))}
    </View>
  )
}

function TagList({ items }: { items: string[] }) {
  return (
    <View style={styles.tagRow}>
      {items.slice(0, 5).map((item, index) => (
        <Text key={`${item}-${index}`} style={styles.tag}>
          {item}
        </Text>
      ))}
    </View>
  )
}

export function RfpPdfDocument({ rfp }: { rfp: RfpDocument }) {
  return (
    <Document title={`${rfp.projectName} RFP`}>
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>제품 제안요청서 (RFP)</Text>
          <Text style={styles.subtitle}>{rfp.projectName}</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>1. 프로젝트 개요</Text>
          <Field label="프로젝트명" value={rfp.projectName} />
          <Field label="제품 한 줄 정의" value={rfp.oneLineDefinition} />
          <Field label="프로젝트 목표" value={rfp.projectGoal} />
          <Field label="최종 활용 목적" value={rfp.finalPurpose} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>2. 타겟 사용자</Text>
          <Field label="메인 타겟" value={rfp.mainTarget} />
          <Field label="사용 상황(TPO)" value={rfp.usageContext} />
          <Field label="핵심 니즈 / 문제" value={rfp.coreNeeds} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>3. 제품 방향</Text>
          <Field label="핵심 가치" value={rfp.coreValue} />
          <Field label="스타일 키워드" value={rfp.styleKeywords.join(', ')} />
          <Field label="피해야 하는 방향" value={rfp.avoidDirections.join(', ')} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>4. 기능 요구사항</Text>
          <Text style={styles.label}>반드시 포함할 핵심 기능</Text>
          <BulletList items={rfp.mustHaveFeatures} />
          <Text style={styles.label}>있으면 좋은 기능</Text>
          <BulletList items={rfp.niceToHaveFeatures} />
          <Text style={styles.label}>이번 범위에서 제외하는 기능</Text>
          <BulletList items={rfp.excludedFeatures} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>5. 구현 및 제작 조건</Text>
          <Field label="예상 예산 범위" value={rfp.budgetRange} />
          <Field label="목표 기간" value={rfp.timeline} />
          <Field label="예상 크기 / 형태 조건" value={rfp.sizeOrForm} />
          <Text style={styles.label}>구현 시 주의할 점</Text>
          <BulletList items={rfp.implementationNotes} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>6. 레퍼런스 및 시장 인사이트</Text>
          <Field label="참고 이미지/레퍼런스 요약" value={rfp.referenceSummary} />
          <Text style={styles.label}>리서치 핵심 인사이트</Text>
          <BulletList items={rfp.researchInsights} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>7. 성공 기준</Text>
          <BulletList items={rfp.successCriteria} />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>8. 다음 액션</Text>
          <BulletList items={rfp.nextActions} />
        </View>
      </Page>
    </Document>
  )
}

export function OnePageRfpPdfDocument({
  rfp,
  selectedReferenceImage,
}: {
  rfp: RfpDocument
  selectedReferenceImage?: string | null
}) {
  return (
    <Document title={`${rfp.projectName} One Page RFP`}>
      <Page size="A4" style={styles.onePage}>
        <View style={styles.onePageHeader}>
          <Text style={styles.onePageTitle}>One Page Design RFP</Text>
          <Text style={styles.onePageSubtitle}>
            {rfp.projectName} · 선택 레퍼런스와 디자인 방향 요약
          </Text>
        </View>

        <View style={styles.onePageGrid}>
          <View style={styles.onePageLeft}>
            {selectedReferenceImage ? (
              <Image src={selectedReferenceImage} style={styles.heroImage} />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.onePageSubtitle}>선택된 레퍼런스 이미지 없음</Text>
              </View>
            )}

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>스타일 키워드</Text>
              <TagList items={rfp.styleKeywords} />
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>레퍼런스 요약</Text>
              <Text style={styles.compactValue}>{rfp.referenceSummary}</Text>
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>제외 방향</Text>
              <CompactBulletList items={rfp.avoidDirections} />
            </View>
          </View>

          <View style={styles.onePageRight}>
            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>프로젝트 개요</Text>
              <CompactField label="제품 정의" value={rfp.oneLineDefinition} />
              <CompactField label="목표" value={rfp.projectGoal} />
              <CompactField label="활용 목적" value={rfp.finalPurpose} />
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>타겟과 핵심 가치</Text>
              <CompactField label="메인 타겟" value={rfp.mainTarget} />
              <CompactField label="사용 상황" value={rfp.usageContext} />
              <CompactField label="핵심 니즈" value={rfp.coreNeeds} />
              <CompactField label="핵심 가치" value={rfp.coreValue} />
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>디자인 아이디어</Text>
              <CompactField label="형태/크기 조건" value={rfp.sizeOrForm} />
              <Text style={styles.compactLabel}>필수 기능</Text>
              <CompactBulletList items={rfp.mustHaveFeatures} />
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>제작 조건</Text>
              <CompactField label="예산" value={rfp.budgetRange} />
              <CompactField label="기간" value={rfp.timeline} />
              <Text style={styles.compactLabel}>구현 주의점</Text>
              <CompactBulletList items={rfp.implementationNotes} />
            </View>

            <View style={styles.compactSection}>
              <Text style={styles.compactSectionTitle}>완료 기준 / 다음 액션</Text>
              <CompactBulletList
                items={[...rfp.successCriteria.slice(0, 2), ...rfp.nextActions.slice(0, 2)]}
              />
            </View>
          </View>
        </View>
      </Page>
    </Document>
  )
}

function normalizeRfpLines(content: string) {
  return content
    .split('\n')
    .map((line) => line.replace(/\r/g, '').trimEnd())
    .filter((line, index, arr) => !(line === '' && arr[index - 1] === ''))
}

export function RawRfpPdfDocument({
  title,
  content,
}: {
  title: string
  content: string
}) {
  const lines = normalizeRfpLines(content)

  return (
    <Document title={`${title} RFP`}>
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>제품 제안요청서 (RFP)</Text>
          <Text style={styles.subtitle}>{title}</Text>
        </View>

        <View style={styles.section}>
          {lines.map((line, index) => {
            const trimmed = line.trim()

            if (!trimmed) {
              return <Text key={`blank-${index}`}> </Text>
            }

            if (trimmed.startsWith('# ')) {
              return (
                <Text key={`h1-${index}`} style={styles.sectionTitle}>
                  {trimmed.slice(2)}
                </Text>
              )
            }

            if (trimmed.startsWith('## ')) {
              return (
                <Text
                  key={`h2-${index}`}
                  style={[styles.sectionTitle, { fontSize: 12, marginTop: 8 }]}
                >
                  {trimmed.slice(3)}
                </Text>
              )
            }

            if (trimmed.startsWith('- ')) {
              return (
                <Text key={`bullet-${index}`} style={styles.bullet}>
                  • {trimmed.slice(2)}
                </Text>
              )
            }

            return (
              <Text key={`p-${index}`} style={[styles.value, { marginBottom: 4 }]}>
                {trimmed}
              </Text>
            )
          })}
        </View>
      </Page>
    </Document>
  )
}
