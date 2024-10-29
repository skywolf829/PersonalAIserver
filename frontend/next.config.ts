/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['lucide-react'],
  output: 'export',  // Enable static exports
  basePath: '/PersonalAIserver', // Replace with your repo name
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig