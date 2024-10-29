/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['lucide-react'],
  output: 'export',  // Enable static exports
  basePath: process.env.NODE_ENV === 'production' ? '/PersonalAIserver' : '',
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig