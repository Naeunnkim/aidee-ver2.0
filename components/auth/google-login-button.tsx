'use client'

import Image from 'next/image'
import { useState, useTransition } from 'react'

import { createClient } from '@/lib/supabase/client'

type GoogleLoginButtonProps = {
  label?: string
  icon?: 'google' | 'naver' | 'kakao'
  disabled?: boolean
  onClick?: () => void
  next?: string
}

export function GoogleLoginButton({
  label = 'Google로 계속하기',
  icon = 'google',
  disabled = false,
  onClick,
  next = '/dashboard',
}: GoogleLoginButtonProps) {
  const [isPending, startTransition] = useTransition()
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const iconStyles = {
    google: 'bg-white text-zinc-700 ring-1 ring-zinc-200',
    naver: 'bg-[#03c75a] text-white',
    kakao: 'bg-[#fee500] text-[#191919]',
  }

  const iconLabel = {
    google: 'G',
    naver: 'N',
    kakao: 'K',
  }

  const isDisabled = disabled || isPending

  function handleClick() {
    if (onClick) {
      onClick()
      return
    }

    if (icon !== 'google') {
      return
    }

    startTransition(async () => {
      setErrorMessage(null)

      const supabase = createClient()
      const redirectTo = `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo,
          queryParams: {
            prompt: 'select_account',
          },
        },
      })

      if (error) {
        setErrorMessage('Google 로그인 시작에 실패했습니다.')
      }
    })
  }

  return (
    <div className="w-full">
      <button
        type="button"
        onClick={handleClick}
        disabled={isDisabled}
        className="inline-flex h-14 w-full items-center justify-center gap-2 rounded-full border border-zinc-200 bg-white px-5 text-base font-semibold text-zinc-900 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {icon === 'google' ? (
          <Image
            src="/icons/social/google.svg"
            alt=""
            width={24}
            height={24}
            unoptimized
            className="h-6 w-6"
          />
        ) : (
          <span
            className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${iconStyles[icon]}`}
          >
            {iconLabel[icon]}
          </span>
        )}
        <span>{isPending && icon === 'google' ? '이동 중...' : label}</span>
      </button>

      {errorMessage ? (
        <p className="mt-2 text-sm text-red-600">{errorMessage}</p>
      ) : null}
    </div>
  )
}
