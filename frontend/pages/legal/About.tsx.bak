import React from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '../../components/branding/Logo';

const About: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Hero Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-cyan-900/30 via-gray-900 to-emerald-900/30">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold mb-6">About Base Gas Optimizer</h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Making Base network more accessible and affordable through AI-powered gas optimization
          </p>
        </div>
      </section>

      <div className="max-w-4xl mx-auto px-8 py-12">
        {/* Mission */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-4">Our Mission</h2>
          <p className="text-lg text-gray-300">
            We believe blockchain technology should be accessible to everyone. High and unpredictable
            gas fees create a barrier to entry for many users. Our mission is to democratize access
            to the Base network by helping users optimize their transaction costs through machine
            learning and data science.
          </p>
        </section>

        {/* Story */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-4">Our Story</h2>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              Base Gas Optimizer was founded in December 2024 during the Coinbase x Queen Mary
              hackathon. As active Base network users ourselves, we experienced firsthand the
              frustration of overpaying for gas fees.
            </p>
            <p className="text-gray-300 mb-4">
              We realized that while gas prices fluctuate significantly throughout the day, most
              users had no way to know when to transact for optimal pricing. Manual timing was
              impractical and ineffective.
            </p>
            <p className="text-gray-300">
              Combining our expertise in machine learning and blockchain development, we built a
              solution that analyzes historical patterns and predicts optimal transaction times.
              What started as a hackathon project has evolved into a tool trusted by thousands of
              Base users.
            </p>
          </div>
        </section>

        {/* Technology */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-4">Our Technology</h2>
          <div className="space-y-4">
            <div className="bg-gray-800 border border-cyan-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-cyan-400 mb-2">🤖 Machine Learning</h3>
              <p className="text-gray-300">
                We use Random Forest regression trained on 30 days of Base network data, achieving
                82% accuracy (R² score). Our models analyze time-of-day patterns, day-of-week trends,
                and network congestion indicators.
              </p>
            </div>

            <div className="bg-gray-800 border border-emerald-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-emerald-400 mb-2">📊 Real-time Data</h3>
              <p className="text-gray-300">
                We collect Base gas prices every 5 minutes, capturing base fees, priority fees, and
                block data. This gives us a comprehensive view of network dynamics.
              </p>
            </div>

            <div className="bg-gray-800 border border-amber-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-amber-400 mb-2">🔄 Continuous Improvement</h3>
              <p className="text-gray-300">
                Our models retrain daily with new data, continuously improving predictions. We track
                accuracy metrics and adjust our approach based on real-world performance.
              </p>
            </div>
          </div>
        </section>

        {/* Values */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Our Values</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-3">🎯</div>
              <h3 className="text-lg font-bold mb-2">Accuracy</h3>
              <p className="text-gray-400 text-sm">
                We're committed to providing the most accurate predictions possible
              </p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">🔒</div>
              <h3 className="text-lg font-bold mb-2">Security</h3>
              <p className="text-gray-400 text-sm">
                Your privacy and security are our top priorities
              </p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">🤝</div>
              <h3 className="text-lg font-bold mb-2">Transparency</h3>
              <p className="text-gray-400 text-sm">
                We're open about our methods, accuracy, and limitations
              </p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="text-center py-12">
          <h2 className="text-3xl font-bold mb-4">Join Us in Making Base More Affordable</h2>
          <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
            Whether you're a DeFi trader, NFT creator, or casual user, Base Gas Optimizer can help
            you save money on every transaction.
          </p>
          <Link
            to="/app"
            className="inline-block px-8 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-lg font-semibold hover:shadow-xl transition transform hover:scale-105"
          >
            Get Started Free →
          </Link>
        </section>
      </div>
    </div>
  );
};

export default About;

