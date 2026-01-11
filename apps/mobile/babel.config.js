module.exports = function (api) {
  api.cache(true);

  const isTest = process.env.NODE_ENV === 'test';

  const plugins = [
    [
      'module-resolver',
      {
        root: ['./'],
        alias: {
          '@': './src',
        },
      },
    ],
  ];

  // Only add reanimated plugin in non-test environments
  if (!isTest) {
    plugins.push('react-native-reanimated/plugin');
  }

  return {
    presets: ['babel-preset-expo'],
    plugins,
  };
};
