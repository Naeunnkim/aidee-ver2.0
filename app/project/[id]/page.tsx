import { notFound, redirect } from 'next/navigation'

import ChatPage from '@/components/project/chat-page'
import { createClient } from '@/lib/supabase/server'

type ProjectPageProps = {
  params: Promise<{
    id: string
  }>
}

export default async function ProjectPage({ params }: ProjectPageProps) {
  const { id } = await params
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect(`/login?next=/project/${id}`)
  }

  const { data: project, error } = await supabase
    .from('projects')
    .select('id, title, status, requirements, created_at')
    .eq('id', id)
    .eq('user_id', user.id)
    .single()

  if (error || !project) {
    notFound()
  }

  const displayName =
    user.user_metadata.full_name ??
    user.user_metadata.name ??
    user.email?.split('@')[0] ??
    '사용자'

  return (
    <ChatPage
      projectId={project.id}
      projectTitle={project.title}
      userName={displayName}
    />
  )
}
