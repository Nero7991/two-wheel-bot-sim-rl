import { defineConfig } from 'vite';

export default defineConfig({
  // Server configuration for development
  server: {
    port: 3000,
    host: true, // Allow external connections
    cors: true,
    headers: {
      // Required headers for WebGPU and SharedArrayBuffer
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },

  // Build configuration
  build: {
    target: 'es2022', // Modern target for WebGPU support
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    
    // Optimize for ML workloads
    rollupOptions: {
      output: {
        // Separate chunks for better caching (will be configured when modules are implemented)
        manualChunks: undefined, // Disable manual chunks for now
        
        // Chunk configuration for future use:
        // manualChunks: {
        //   'ml-network': ['./src/network'],
        //   'ml-training': ['./src/training'],
        //   'physics-engine': ['./src/physics'],
        //   'visualization': ['./src/visualization'],
        //   'export-utils': ['./src/export'],
        // },
        
        // Optimize chunk size for ML workloads
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
      
      // External dependencies that should not be bundled
      external: [],
    },
    
    // Increase chunk size warning limit for ML models
    chunkSizeWarningLimit: 2000, // 2MB instead of default 500KB
    
    // Minification settings
    minify: 'terser',
    terserOptions: {
      compress: {
        // Preserve WebGPU and ML-related code
        keep_fnames: true,
        keep_classnames: true,
        
        // Remove console logs in production
        drop_console: true,
        drop_debugger: true,
      },
      mangle: {
        // Preserve WebGPU API calls
        keep_fnames: true,
        keep_classnames: true,
      },
    },
  },

  // Plugin configuration
  plugins: [
    // Custom plugin for WebGPU headers
    {
      name: 'webgpu-headers',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // Add headers required for WebGPU
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          next();
        });
      },
    },
  ],

  // Optimization settings
  optimizeDeps: {
    // Include dependencies that should be pre-bundled
    include: [],
    
    // Exclude large ML libraries from pre-bundling
    exclude: [],
    
    // ESBuild options for dependency optimization
    esbuildOptions: {
      target: 'es2022',
      
      // Preserve WebGPU-related names
      keepNames: true,
    },
  },

  // Define global constants
  define: {
    __DEV__: JSON.stringify(process.env.NODE_ENV === 'development'),
    __WEBGPU_ENABLED__: 'true',
    __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
  },

  // Worker configuration for ML computations
  worker: {
    format: 'es',
    plugins: () => [],
  },

  // CSS configuration
  css: {
    devSourcemap: true,
    preprocessorOptions: {
      // Add any CSS preprocessor options here
    },
  },

  // Asset handling
  assetsInclude: [
    // Include ML model files as assets
    '**/*.onnx',
    '**/*.bin',
    '**/*.weights',
    '**/*.json',
  ],

  // Preview server configuration (for vite preview)
  preview: {
    port: 4173,
    host: true,
    cors: true,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },

  // ESBuild configuration
  esbuild: {
    target: 'es2022',
    keepNames: true, // Important for WebGPU debugging
    legalComments: 'none',
  },

  // Resolve configuration
  resolve: {
    alias: {
      // Create convenient aliases for module imports
      '@': '/src',
      '@network': '/src/network',
      '@physics': '/src/physics',
      '@training': '/src/training',
      '@visualization': '/src/visualization',
      '@export': '/src/export',
    },
    extensions: ['.js', '.ts', '.json', '.wasm'],
  },

  // Experimental features
  experimental: {
    // Enable any experimental features needed for WebGPU
  },
});