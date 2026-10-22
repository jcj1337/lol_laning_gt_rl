import { createContext, useContext, useSyncExternalStore } from 'react';
import type { AppStore } from '../game/store';

export const StoreContext = createContext<AppStore | null>(null);

/** The store, re-rendering the component on every change. */
export function useStore(): AppStore {
  const store = useContext(StoreContext);
  if (!store) throw new Error('no store');
  useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  return store;
}
