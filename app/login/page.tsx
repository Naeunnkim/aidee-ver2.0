import { GoogleLoginButton } from '@/components/auth/google-login-button'

export default function LoginPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-zinc-50 px-6 py-10 text-zinc-950">
      <div className="mx-auto flex min-h-[calc(100vh-5rem)] max-w-[1440px] items-center justify-center">
        <section className="relative w-full overflow-hidden rounded-[40px] border border-zinc-200/80 bg-white shadow-[0_30px_120px_rgba(15,23,42,0.08)]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,#f8fbff_0%,#f8fafc_42%,#f4f4f5_100%)]" />
          <div className="relative flex min-h-[776px] items-center justify-center px-6 py-16">
            <div className="flex w-full max-w-sm flex-col items-start gap-8">
              <div className="flex w-full flex-col items-start gap-8">
                <div className="flex w-full flex-col items-center gap-5">
                  <div className="flex w-full flex-col items-center gap-6">
                    <div className="text-center text-base font-medium leading-6 text-zinc-700">
                      제품 디자인 전문 AI 멀티 에이전트 플랫폼
                    </div>
                  </div>
                </div>

                <div className="inline-flex w-full items-center justify-center gap-2">
                  <div className="h-px flex-1 bg-zinc-200" />
                  <div className="text-sm font-medium leading-5 text-zinc-700">
                    또는
                  </div>
                  <div className="h-px flex-1 bg-zinc-200" />
                </div>
              </div>

              <div className="flex w-full flex-col items-start gap-3">
                <GoogleLoginButton />
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

              <p className="text-sm leading-6 text-zinc-500">
                지금은 Google 로그인부터 연결하는 게 가장 깔끔합니다. 네이버와
                카카오는 UI만 먼저 둔 상태입니다.
              </p>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
