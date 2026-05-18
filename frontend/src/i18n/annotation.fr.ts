/**
 * French strings for the annotation page.
 * Usage: import { t } from '../i18n/annotation.fr';
 *        t('save.localSuccess')  → "Annotations enregistrées localement"
 *        t('save.internalError', { traceId: 'a1b2' }) → "Erreur interne du serveur (id=a1b2)"
 */

export const annotationFr = {
  save: {
    localSuccess: 'Annotations enregistrées localement',
    remoteSuccess: 'Annotations envoyées au serveur',
    permissionDenied: 'Sauvegarde impossible : droits insuffisants sur le dossier annotations',
    diskFull: 'Sauvegarde impossible : espace disque insuffisant',
    storageError: 'Erreur de stockage : {message}',
    internalError: 'Erreur interne du serveur (id={traceId})',
    networkError: 'Impossible de joindre le serveur',
  },
  pan: {
    enabledHint: 'Glissez pour vous déplacer (zoom actif)',
    resetView: 'Réinitialiser la vue',
  },
  mode: {
    annotate: 'Mode annotation',
    view: 'Mode visualisation',
  },
} as const;

type DeepKeys<T, Prefix extends string = ''> = {
  [K in keyof T]: T[K] extends Record<string, unknown>
    ? DeepKeys<T[K], `${Prefix}${K & string}.`>
    : `${Prefix}${K & string}`;
}[keyof T];

type AnnotationFrKey = DeepKeys<typeof annotationFr>;

/**
 * Get a French string by dot-notation key, with optional {var} substitution.
 */
export function t(key: AnnotationFrKey, vars?: Record<string, string>): string {
  const parts = key.split('.');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let value: any = annotationFr;
  for (const part of parts) {
    value = value?.[part];
  }
  if (typeof value !== 'string') return key;
  if (!vars) return value;
  return value.replace(/\{(\w+)\}/g, (_, name) => vars[name] ?? `{${name}}`);
}
