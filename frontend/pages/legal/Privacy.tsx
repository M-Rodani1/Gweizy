import React from 'react';
import { Link } from 'react-router-dom';

const Privacy: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center space-x-2 mb-8">
          <Link to="/" className="text-xl font-bold text-white hover:text-cyan-400 transition">
            Base Gas Optimizer
          </Link>
        </div>
        
        <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>
        <p className="text-gray-400 mb-8">Last updated: December 13, 2024</p>

        <div className="space-y-6 text-gray-300">
          <section>
            <h2 className="text-2xl font-bold text-white mb-3">1. Information We Collect</h2>
            
            <h3 className="text-xl font-semibold text-cyan-400 mt-4 mb-2">Information You Provide:</h3>
            <ul className="list-disc ml-6 space-y-1">
              <li>Email address (for account creation and notifications)</li>
              <li>Wallet address (if you choose to connect a wallet)</li>
              <li>Usage preferences and alert settings</li>
            </ul>

            <h3 className="text-xl font-semibold text-cyan-400 mt-4 mb-2">Information We Collect Automatically:</h3>
            <ul className="list-disc ml-6 space-y-1">
              <li>Device information (browser type, operating system)</li>
              <li>Usage data (features accessed, time spent)</li>
              <li>IP address and general location</li>
              <li>Cookies and similar technologies</li>
            </ul>

            <h3 className="text-xl font-semibold text-cyan-400 mt-4 mb-2">Blockchain Data:</h3>
            <ul className="list-disc ml-6 space-y-1">
              <li>Publicly available transaction history (if wallet connected)</li>
              <li>Gas prices paid and transaction timing</li>
              <li>This data is public on the Base blockchain</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">2. How We Use Your Information</h2>
            <p>We use collected information to:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Provide and improve our gas price prediction Service</li>
              <li>Send you alerts and notifications (if enabled)</li>
              <li>Analyze usage patterns to improve our models</li>
              <li>Communicate with you about your account</li>
              <li>Comply with legal obligations</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">3. Information Sharing</h2>
            <p className="mb-2">
              <strong className="text-green-400">We do NOT sell your personal data.</strong>
            </p>
            <p>We may share information with:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Service providers (hosting, analytics, email)</li>
              <li>Legal authorities (if required by law)</li>
              <li>Business partners (with your consent)</li>
            </ul>
            <p className="mt-2">
              <strong>We never share:</strong> Your private keys, passwords, or financial account details
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">4. Data Security</h2>
            <p>We implement security measures including:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Encryption of data in transit (HTTPS/TLS)</li>
              <li>Encrypted storage of sensitive information</li>
              <li>Regular security audits and updates</li>
              <li>Access controls and authentication</li>
            </ul>
            <p className="mt-2 text-amber-400">
              <strong>Important:</strong> No method of transmission over the internet is 100% secure.
              While we strive to protect your data, we cannot guarantee absolute security.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">5. Your Rights (GDPR Compliance)</h2>
            <p>You have the right to:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li><strong>Access:</strong> Request a copy of your personal data</li>
              <li><strong>Rectification:</strong> Correct inaccurate data</li>
              <li><strong>Erasure:</strong> Request deletion of your data</li>
              <li><strong>Portability:</strong> Receive your data in a portable format</li>
              <li><strong>Object:</strong> Object to certain data processing</li>
              <li><strong>Withdraw Consent:</strong> Withdraw consent at any time</li>
            </ul>
            <p className="mt-2">
              To exercise these rights, email us at{' '}
              <a href="mailto:privacy@basegasoptimizer.com" className="text-cyan-400">
                privacy@basegasoptimizer.com
              </a>
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">6. Cookies</h2>
            <p>We use cookies for:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Essential functionality (authentication, preferences)</li>
              <li>Analytics (understanding how you use our Service)</li>
              <li>Performance optimization</li>
            </ul>
            <p className="mt-2">
              You can disable cookies in your browser settings, but some features may not work properly.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">7. Third-Party Services</h2>
            <p>Our Service may contain links to third-party websites or services:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>BaseScan (blockchain explorer)</li>
              <li>Wallet providers (MetaMask, Coinbase Wallet)</li>
              <li>Analytics providers (Google Analytics)</li>
            </ul>
            <p className="mt-2">
              We are not responsible for the privacy practices of these third parties.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">8. Children's Privacy</h2>
            <p>
              Our Service is not intended for children under 18. We do not knowingly collect data
              from children. If you believe we have collected data from a child, please contact us
              immediately.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">9. Changes to Privacy Policy</h2>
            <p>
              We may update this Privacy Policy from time to time. We will notify you of significant
              changes via email or through the Service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">10. Contact Us</h2>
            <p>
              For privacy questions or to exercise your rights, contact us at:
            </p>
            <div className="mt-2 bg-gray-800 p-4 rounded-lg">
              <p>Email: <a href="mailto:privacy@basegasoptimizer.com" className="text-cyan-400">privacy@basegasoptimizer.com</a></p>
            </div>
          </section>
        </div>

        <div className="mt-12 p-6 bg-gray-800 border border-gray-700 rounded-lg">
          <h3 className="text-lg font-bold text-white mb-2">🔒 Your Privacy Matters</h3>
          <p className="text-sm text-gray-400">
            We are committed to protecting your privacy. We never sell your data, never access your
            private keys, and only collect information necessary to provide our Service.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Privacy;

