import { redirect } from 'next/navigation'

import { ProjectList } from '@/components/dashboard/project-list'
import { createClient } from '@/lib/supabase/server'

export default async function DashboardPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect('/login?next=/dashboard')
  }

  const displayName =
    user.user_metadata.full_name ??
    user.user_metadata.name ??
    user.email?.split('@')[0] ??
    '사용자'

  return (
    <ProjectList
      user={{
        displayName,
        email: user.email ?? '',
        id: user.id,
      }}
    />
  )
}
