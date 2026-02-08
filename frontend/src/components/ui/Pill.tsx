/**
 * @deprecated Use Badge component instead. Pill will be removed in a future version.
 *
 * Migration guide:
 * - color="cyan" -> variant="accent"
 * - color="green" -> variant="success"
 * - color="yellow" -> variant="warning"
 * - color="gray" -> variant="neutral"
 */
import { Badge } from './Badge';

export { Badge as Pill };
export default Badge;
