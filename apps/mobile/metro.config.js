// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add react-dom shim for packages that incorrectly import it
config.resolver.resolveRequest = (context, moduleName, platform) => {
  // Redirect react-dom to react-native
  if (moduleName === 'react-dom') {
    return {
      filePath: require.resolve('./src/shims/react-dom.js'),
      type: 'sourceFile',
    };
  }
  // Fall back to default resolution
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
