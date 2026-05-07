import Image from 'next/image'
import { redirect } from 'next/navigation'

import { GoogleLoginButton } from '@/components/auth/google-login-button'
import { createClient } from '@/lib/supabase/server'

function getSafeNextPath(next: string | string[] | undefined) {
  if (typeof next !== 'string' || !next.startsWith('/')) {
    return '/dashboard'
  }

  return next.startsWith('//') ? '/dashboard' : next
}

type LoginPageProps = {
  searchParams: Promise<{
    next?: string | string[]
  }>
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const [{ next }, supabase] = await Promise.all([searchParams, createClient()])
  const safeNextPath = getSafeNextPath(next)
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect(safeNextPath)
  }

  return (
    <main className="h-[100svh] overflow-hidden bg-[#F8F9FA] px-4 py-4 text-zinc-950 sm:px-6 sm:py-6 lg:py-8">
      <div className="mx-auto flex h-full max-w-[1440px] items-center justify-center">
        <section className="flex w-full justify-center px-5 py-8 sm:px-6 sm:py-10">
          <div className="flex w-full max-w-sm flex-col items-start gap-6 sm:gap-8">
            <div className="flex w-full flex-col items-start gap-6 sm:gap-8">
              <div className="flex w-full flex-col items-center gap-5">
                <div className="flex w-full flex-col items-center gap-6">
                  <Image
                    src="/brand/aidee-logo-blue.svg"
                    alt="Aidee"
                    width={115}
                    height={40}
                    unoptimized
                    priority
                    className="h-10 w-auto"
                  />
                  <div className="text-center text-base font-medium leading-6 text-zinc-700">
                    제품 디자인 전문 AI 멀티 에이전트 플랫폼
                  </div>
                </div>
              </div>

              <div className="inline-flex w-full items-center justify-center gap-2">
                <div className="h-px flex-1 bg-zinc-200" />
                <div className="text-sm font-medium leading-5 text-zinc-700">
                  시작하기
                </div>
                <div className="h-px flex-1 bg-zinc-200" />
              </div>
            </div>

            <div className="flex w-full flex-col items-start gap-3">
              <GoogleLoginButton next={safeNextPath} />
              <GoogleLoginButton
                label="네이버로 계속하기"
                icon="naver"
                disabled
              />
              <GoogleLoginButton
                label="카카오로 계속하기"
                icon="kakao"
                disabled
              />
            </div>

            <p className="text-center text-sm leading-6 text-zinc-500">
              현재 베타 환경에서는 안정적인 인증 테스트를 위해 Google 로그인만 활성화되어 있습니다.
              네이버 및 카카오 로그인은 향후 업데이트를 통해 순차적으로 지원될 예정입니다.
            </p>
          </div>
        </section>
      </div>
    </main>
  )
}
