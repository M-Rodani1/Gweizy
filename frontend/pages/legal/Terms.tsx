import React from 'react';
import { Link } from 'react-router-dom';

const Terms: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center space-x-2 mb-8">
          <Link to="/" className="text-xl font-bold text-white hover:text-cyan-400 transition">
            Base Gas Optimizer
          </Link>
        </div>
        
        <h1 className="text-4xl font-bold mb-8">Terms of Service</h1>
        <p className="text-gray-400 mb-8">Last updated: December 13, 2024</p>

        <div className="space-y-6 text-gray-300">
          <section>
            <h2 className="text-2xl font-bold text-white mb-3">1. Acceptance of Terms</h2>
            <p>
              By accessing or using Base Gas Optimizer ("Service"), you agree to be bound by these
              Terms of Service. If you do not agree to these terms, please do not use our Service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">2. Description of Service</h2>
            <p>
              Base Gas Optimizer provides machine learning-powered predictions of Base network gas
              prices. Our Service offers:
            </p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Gas price predictions for 1 hour, 4 hours, and 24 hours ahead</li>
              <li>Price alerts and notifications</li>
              <li>Transaction history analysis (for connected wallets)</li>
              <li>Savings calculations and recommendations</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">3. Disclaimer of Accuracy</h2>
            <p className="mb-2">
              <strong className="text-amber-400">IMPORTANT:</strong> Our predictions are estimates
              based on historical data and machine learning models. We make no guarantees about:
            </p>
            <ul className="list-disc ml-6 space-y-1">
              <li>The accuracy of any prediction</li>
              <li>The amount of money you will save</li>
              <li>The optimal timing of your transactions</li>
            </ul>
            <p className="mt-2">
              Gas prices on blockchain networks are inherently unpredictable. You are solely
              responsible for your transaction timing decisions.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">4. Wallet Connection</h2>
            <p>
              When you connect your wallet to our Service:
            </p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>We never access your private keys</li>
              <li>We only read publicly available blockchain data</li>
              <li>We do not request signing permissions or transaction approvals</li>
              <li>You can disconnect your wallet at any time</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">5. User Responsibilities</h2>
            <p>You agree to:</p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Use the Service in compliance with all applicable laws</li>
              <li>Not attempt to manipulate or interfere with our predictions</li>
              <li>Not use the Service for any illegal or unauthorized purpose</li>
              <li>Keep your account credentials secure</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">6. Subscription and Payment</h2>
            <p>
              For paid subscription plans:
            </p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Subscriptions are billed monthly in advance</li>
              <li>You can cancel your subscription at any time</li>
              <li>Refunds are provided on a case-by-case basis</li>
              <li>Free trial users are not charged until the trial period ends</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">7. Limitation of Liability</h2>
            <p>
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, BASE GAS OPTIMIZER SHALL NOT BE LIABLE FOR:
            </p>
            <ul className="list-disc ml-6 mt-2 space-y-1">
              <li>Any financial losses resulting from inaccurate predictions</li>
              <li>Lost profits or business opportunities</li>
              <li>Service interruptions or data loss</li>
              <li>Third-party actions or blockchain network issues</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">8. Termination</h2>
            <p>
              We reserve the right to terminate or suspend your access to the Service at any time,
              without notice, for conduct that we believe violates these Terms or is harmful to
              other users, us, or third parties, or for any other reason.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">9. Changes to Terms</h2>
            <p>
              We may update these Terms from time to time. Continued use of the Service after
              changes constitutes acceptance of the new Terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-white mb-3">10. Contact</h2>
            <p>
              For questions about these Terms, please contact us at:{' '}
              <a href="mailto:legal@basegasoptimizer.com" className="text-cyan-400 hover:text-cyan-300">
                legal@basegasoptimizer.com
              </a>
            </p>
          </section>
        </div>

        <div className="mt-12 p-6 bg-gray-800 border border-gray-700 rounded-lg">
          <p className="text-sm text-gray-400">
            By using Base Gas Optimizer, you acknowledge that you have read, understood, and agree
            to be bound by these Terms of Service.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Terms;

