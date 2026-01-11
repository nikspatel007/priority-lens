/**
 * React DOM shim for React Native
 *
 * Some packages (like @clerk/clerk-react) incorrectly import react-dom
 * even when running in React Native. This shim provides no-op implementations
 * of the commonly used react-dom APIs.
 */

// createPortal - returns the children directly (no portal in RN)
export function createPortal(children, _container) {
  return children;
}

// flushSync - executes callback immediately
export function flushSync(callback) {
  callback();
}

// render - no-op
export function render() {}

// unmountComponentAtNode - no-op
export function unmountComponentAtNode() {}

// findDOMNode - returns null
export function findDOMNode() {
  return null;
}

// hydrate - no-op
export function hydrate() {}

// createRoot - returns a mock root
export function createRoot() {
  return {
    render: () => {},
    unmount: () => {},
  };
}

// hydrateRoot - returns a mock root
export function hydrateRoot() {
  return {
    render: () => {},
    unmount: () => {},
  };
}

export default {
  createPortal,
  flushSync,
  render,
  unmountComponentAtNode,
  findDOMNode,
  hydrate,
  createRoot,
  hydrateRoot,
};
