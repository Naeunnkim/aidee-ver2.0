export function getSupabaseUrl() {
  const value = process.env.NEXT_PUBLIC_SUPABASE_URL

  if (!value) {
    throw new Error('Missing Supabase environment variable: NEXT_PUBLIC_SUPABASE_URL')
  }

  return value
}

export function getSupabasePublishableKey() {
  const value =
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!value) {
    throw new Error(
      'Missing Supabase environment variable: NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY (or NEXT_PUBLIC_SUPABASE_ANON_KEY)'
    )
  }

  return value
}
