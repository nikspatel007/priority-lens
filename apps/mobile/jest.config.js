/** @type {import('jest').Config} */
module.exports = {
  setupFiles: ['<rootDir>/jest.globals.js'],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testMatch: ['**/__tests__/**/*.test.ts?(x)', '**/*.test.ts?(x)'],
  testPathIgnorePatterns: ['/node_modules/', '/e2e/'],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/__tests__/**',
    '!src/mocks/**',
    '!src/types/**',
    '!src/**/index.ts',
    '!src/sdui/types.ts',
  ],
  coverageThreshold: {
    global: {
      // Note: statements and functions are set to 99% due to istanbul quirks with useCallback
      // We maintain 100% line and branch coverage
      statements: 99,
      branches: 100,
      functions: 99,
      lines: 100,
    },
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(png|jpg|jpeg|gif|svg)$': '<rootDir>/src/mocks/fileMock.js',
  },
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|expo|@expo|expo-.*|@unimodules|unimodules|react-navigation|@react-navigation|native-base|react-native-.*|@clerk|@livekit|livekit-client)/)',
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  testEnvironment: 'node',
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,
};
